from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import os
import random
import re
import unicodedata
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence, cast

from app.models.advice import Advice, AdviceKind, AdviceRecommendation, AdviceRequestContext
from app.repositories.advice_repository import AdviceRepository, EmbeddingUpdatableAdviceRepository
from app.repositories.category_repository import AdviceCategoryRepository
from app.integrations.openai import (
    OpenAISettings,
    create_async_openai_client,
    get_openai_settings,
)
from app.repositories.user_persona_repository import UserPersonaProvider

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as OpenAIClient  # type: ignore[import]
else:
    OpenAIClient = Any  # type: ignore[misc]

try:  # pragma: no cover - runtime availability check
    from openai import AsyncOpenAI as _RuntimeAsyncOpenAI
except ImportError:
    _RuntimeAsyncOpenAI = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryMatch:
    name: str
    score: float
    rank: int


class AdviceCategoryClassifier(Protocol):
    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        raise NotImplementedError


@dataclass(frozen=True)
class AdviceIntentMatch:
    kind: AdviceKind
    score: float


class AdviceIntentDetector(Protocol):
    async def detect_preferred_kind(
        self, user_message: str
    ) -> AdviceIntentMatch | None:
        raise NotImplementedError


class AdviceResponseGenerator(Protocol):
    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        raise NotImplementedError

    def set_log_sink(self, sink: Callable[[str], None]) -> None:
        ...


class AdviceNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdviceSelectionResult:
    advice: Advice
    categories: Sequence[str]
    preferred_kind: AdviceKind | None


class AdviceSelectionPipeline:
    def __init__(
        self,
        advice_repository: AdviceRepository,
        category_repository: AdviceCategoryRepository,
        category_classifier: AdviceCategoryClassifier,
        intent_detector: AdviceIntentDetector,
        response_generator: AdviceResponseGenerator,
        *,
        max_item_categories: int = 7,
    ) -> None:
        self._advice_repository = advice_repository
        self._category_repository = category_repository
        self._category_classifier = category_classifier
        self._intent_detector = intent_detector
        self._response_generator = response_generator
        self._max_item_categories = max_item_categories
        self._category_frequency_cache: dict[str, int] | None = None
        self._total_advice_count: int = 0
        self._frequency_lock = asyncio.Lock()
        self._latest_events: list[str] = []
        if hasattr(self._response_generator, "set_log_sink"):
            try:
                self._response_generator.set_log_sink(
                    self._record)  # type: ignore[attr-defined]
            except TypeError:
                pass

    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        self._latest_events = []
        category_matches = await self._infer_categories(request.user_message)
        if not category_matches:
            self._record(
                "Nie można wygenerować porady bez choć jednej rozpoznanej kategorii."
            )
            raise AdviceNotFoundError(
                "Brak dopasowanych kategorii do wypowiedzi użytkownika."
            )
        intent_match = await self._intent_detector.detect_preferred_kind(
            request.user_message
        )
        if intent_match:
            self._record(
                f"Rozpoznano prośbę o rodzaj: {intent_match.kind.value} (score={intent_match.score:.3f})"
            )
        else:
            self._record(
                "Brak jednoznacznej prośby o konkretny rodzaj porady.")

        advice = await self._select_advice(intent_match, category_matches)
        if advice is None:
            raise AdviceNotFoundError(
                "No advice found for the given criteria.")

        chat_response = await self._response_generator.generate_response(
            advice=advice,
            request=request,
            categories=tuple(match.name for match in category_matches),
            preferred_kind=intent_match.kind if intent_match else None,
        )
        return AdviceRecommendation(advice=advice, chat_response=chat_response)

    def get_latest_events(self) -> Sequence[str]:
        return tuple(self._latest_events)

    def _record(self, message: str) -> None:
        self._latest_events.append(message)
        logger.info(message)

    async def _infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        inferred_matches = await self._category_classifier.infer_categories(
            user_message
        )
        if not inferred_matches:
            self._record(
                "Nie wykryto żadnych kategorii w wypowiedzi użytkownika.")
            return ()
        known_categories = await self._category_repository.get_all()
        variant_map = self._build_known_category_map(known_categories)
        unique_categories: list[CategoryMatch] = []
        seen = set()
        for match in inferred_matches:
            normalized = match.name.lower()
            matched_name = self._match_category_to_known(
                normalized, variant_map)
            if matched_name:
                canonical = matched_name
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    unique_categories.append(
                        CategoryMatch(
                            name=canonical,
                            score=match.score,
                            rank=match.rank,
                        )
                    )
                    self._record(
                        f"Wykryta kategoria '{canonical}' | score={match.score:.3f} | pozycja={match.rank}"
                    )
            else:
                self._record(
                    f"Kategoria '{match.name}' nie pasuje do bazy (próbowano wariantów: {self._build_category_variants(normalized)})"
                )
        if not unique_categories:
            self._record(
                "Żadna wykryta kategoria nie istnieje w bazie kategorii.")
        return tuple(unique_categories)

    async def _select_advice(
        self,
        intent_match: AdviceIntentMatch | None,
        matched_categories: Sequence[CategoryMatch],
    ) -> Advice | None:
        candidates: Sequence[Advice] = ()
        category_names = tuple(match.name for match in matched_categories)
        preferred_kind = intent_match.kind if intent_match else None

        if preferred_kind is not None:
            if matched_categories:
                candidates = await self._advice_repository.get_by_kind_and_containing_any_category(
                    preferred_kind,
                    category_names,
                )
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)

        if not candidates:
            if matched_categories:
                all_advice = await self._advice_repository.get_all()
                candidates = tuple(
                    advice
                    for advice in all_advice
                    if self._contains_any_category(advice, category_names)
                )

        if not candidates:
            candidates = await self._advice_repository.get_all()

        await self._ensure_category_frequencies()
        selected = self._rank_candidates(
            candidates,
            matched_categories,
            self._category_frequency_cache or {},
            max(self._total_advice_count, len(candidates)),
            intent_match,
        )
        if selected:
            self._record(
                f"Wybrana porada: {selected.name} ({selected.kind.value}).")
        else:
            self._record(
                "Nie udało się wybrać porady na podstawie dostępnych kandydatów.")
        return selected

    @staticmethod
    def _contains_any_category(advice: Advice, categories: Sequence[str]) -> bool:
        advice_categories = {category.lower()
                             for category in advice.categories}
        for category in categories:
            if category.lower() in advice_categories:
                return True
        return False

    def _rank_candidates(
        self,
        candidates: Sequence[Advice],
        matched_categories: Sequence[CategoryMatch],
        category_frequencies: Mapping[str, int],
        total_advice_count: int,
        intent_match: AdviceIntentMatch | None,
    ) -> Advice | None:
        if not candidates:
            return None

        if not matched_categories:
            if intent_match:
                matching_kind = [
                    advice for advice in candidates if advice.kind == intent_match.kind
                ]
                if matching_kind:
                    self._record(
                        f"Brak dopasowanych kategorii – wybieram losowo spośród porad rodzaju {intent_match.kind.value}."
                    )
                    return random.choice(matching_kind)
            self._record(
                "Brak dopasowanych kategorii – wybieram losowo dowolną poradę."
            )
            return random.choice(tuple(candidates))

        match_map = {match.name.lower(): match for match in matched_categories}
        population: list[Advice] = []
        weights: list[float] = []

        for candidate in candidates:
            candidate_category_set = {
                category.lower() for category in candidate.categories
            }
            applicable_matches = [
                match_map[name]
                for name in candidate_category_set
                if name in match_map
            ]

            if not applicable_matches:
                base_weight = 0.001
                if intent_match and candidate.kind == intent_match.kind:
                    base_weight = 0.001
                population.append(candidate)
                weights.append(base_weight)
                continue

            for match in applicable_matches:
                freq = category_frequencies.get(match.name.lower(), 0)
                if freq == 1 and match.rank == 1:
                    self._record(
                        f"Unikatowa kategoria '{match.name}' (pozycja 1) występuje tylko w poradzie '{candidate.name}'. Wybór deterministyczny."
                    )
                    return candidate

            specificity = (self._max_item_categories + 1) / (
                len(candidate.categories) + 1
            )
            weight = 0.0
            for match in applicable_matches:
                freq = category_frequencies.get(match.name.lower(), 0) or 1
                rarity = ((total_advice_count + 1) / (freq + 1)) ** 2
                if freq == 1 and match.rank == 2:
                    rarity *= 12
                ranking_weight = (
                    len(matched_categories) - match.rank + 1
                ) / len(matched_categories)
                similarity = match.score ** 2
                incremental = similarity * ranking_weight * rarity * specificity
                weight += incremental

            if intent_match:
                if candidate.kind == intent_match.kind:
                    weight *= 1.8
                else:
                    weight *= 0.15

            weight *= random.uniform(0.85, 1.15)
            weights.append(max(weight, 0.02))
            population.append(candidate)

        total_weight = sum(weights)
        if total_weight <= 0:
            self._record(
                "Suma wag wyniosła 0 – wybieram losowo z dostępnych kandydatów."
            )
            return random.choice(tuple(candidates))

        selected = random.choices(
            population=population, weights=weights, k=1)[0]
        selected_index = population.index(selected)
        self._log_weights(
            population,
            weights,
            selected_index,
            matched_categories,
            intent_match,
        )
        return selected

    @staticmethod
    def _match_category_to_known(
        candidate: str, variant_map: dict[str, str]
    ) -> str | None:
        for variant in AdviceSelectionPipeline._build_category_variants(candidate):
            mapped = variant_map.get(variant.lower())
            if mapped:
                return mapped
        return None

    @staticmethod
    def _build_category_variants(candidate: str) -> tuple[str, ...]:
        stripped = candidate.strip().lower()
        variants: list[str] = []
        ascii_variant = (
            unicodedata.normalize("NFKD", stripped)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        possible = [
            stripped,
            ascii_variant,
            stripped.replace(" ", "-"),
            ascii_variant.replace(" ", "-"),
            stripped.replace(" ", "_"),
            ascii_variant.replace(" ", "_"),
        ]
        for variant in possible:
            if variant and variant not in variants:
                variants.append(variant)
            sanitized = "".join(
                ch for ch in variant if ch.isalnum() or ch in {"-", "_", " "}
            ).strip()
            if sanitized and sanitized not in variants:
                variants.append(sanitized)
        return tuple(variants)

    @staticmethod
    def _build_known_category_map(categories: Sequence[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name in categories:
            for variant in AdviceSelectionPipeline._build_category_variants(name):
                key = variant.lower()
                mapping.setdefault(key, name)
        return mapping

    async def _ensure_category_frequencies(self) -> None:
        if self._category_frequency_cache is not None:
            return
        async with self._frequency_lock:
            if self._category_frequency_cache is not None:
                return
            all_advice = await self._advice_repository.get_all()
            counter: Counter[str] = Counter()
            for advice in all_advice:
                seen = set()
                for category in advice.categories:
                    normalized = category.lower()
                    if normalized not in seen:
                        counter[normalized] += 1
                        seen.add(normalized)
            self._category_frequency_cache = dict(counter)
            self._total_advice_count = len(all_advice)
            self._record(
                f"Zaktualizowano częstotliwości kategorii na podstawie {self._total_advice_count} porad."
            )

    def _log_weights(
        self,
        population: Sequence[Advice],
        weights: Sequence[float],
        selected_index: int,
        matched_categories: Sequence[CategoryMatch],
        intent_match: AdviceIntentMatch | None,
    ) -> None:
        lines = ["Podsumowanie wag dla kandydatów:"]
        for idx, (advice, weight) in enumerate(zip(population, weights)):
            marker = " <= WYBRANA" if idx == selected_index else ""
            lines.append(f"  - {advice.name}: waga={weight:.3f}{marker}")
        if matched_categories:
            lines.append(
                "Kategorie podstawowe: "
                + ", ".join(
                    f"{match.name} (rank={match.rank}, score={match.score:.3f})"
                    for match in matched_categories
                )
            )
        else:
            lines.append("Kategorie podstawowe: brak")
        if intent_match:
            lines.append(
                f"Prośba użytkownika: {intent_match.kind.value} (score={intent_match.score:.3f})"
            )
        self._record("\n".join(lines))


class PersonaEmbeddingAdviceSelectionPipeline:
    """
    Alternative selection pipeline that does NOT use categories.

    It builds a rich embedding for the current request based on:
    - psychological profile (required),
    - vocational profile (optional),
    - current user message,

    and compares it with cached embeddings of all advices.
    Intent detection (`kind`) is preserved and used as a hard/soft filter.
    """

    def __init__(
        self,
        advice_repository: EmbeddingUpdatableAdviceRepository,
        intent_detector: AdviceIntentDetector,
        response_generator: AdviceResponseGenerator,
        persona_provider: UserPersonaProvider,
        *,
        similarity_threshold: float = 0.33,
        embeddings_model: str | None = None,
    ) -> None:
        if _RuntimeAsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install it with `pip install openai`."
            )
        self._advice_repository = advice_repository
        self._intent_detector = intent_detector
        self._response_generator = response_generator
        self._persona_provider = persona_provider
        self._similarity_threshold = similarity_threshold

        settings = get_openai_settings()
        runtime_client: OpenAIClient = create_async_openai_client(settings)
        if not isinstance(runtime_client, _RuntimeAsyncOpenAI):
            raise TypeError(
                "client must be an instance of openai.AsyncOpenAI.")
        self._client = runtime_client
        self._embeddings_model = (
            embeddings_model
            or os.getenv("OPENAI_ADVICE_EMBEDDING_MODEL")
            or settings.embeddings_model
        )

        self._latest_events: list[str] = []
        # In-memory cache keyed by advice id
        self._embedding_cache: dict[int, tuple[float, ...]] = {}
        self._cache_max_size = int(
            os.getenv("ADVICE_RESULT_CACHE_SIZE", "20") or 0)
        self._result_cache: OrderedDict[
            tuple[str, str], AdviceRecommendation
        ] = OrderedDict()

        if hasattr(self._response_generator, "set_log_sink"):
            try:
                self._response_generator.set_log_sink(
                    self._record)  # type: ignore[attr-defined]
            except TypeError:
                pass

    def get_latest_events(self) -> Sequence[str]:
        return tuple(self._latest_events)

    def _record(self, message: str) -> None:
        self._latest_events.append(message)
        logger.info(message)

    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        self._latest_events = []
        user_id = request.user_identifier.user_id
        if not user_id:
            self._record(
                "Tryb embeddingowy wymaga identyfikatora użytkownika (user_id)."
            )
            raise AdviceNotFoundError(
                "Brak identyfikatora użytkownika – nie można wykorzystać profilów testowych."
            )

        psych_profile = await self._persona_provider.get_persona_by_type(
            user_id, "psychology"
        )
        if not psych_profile:
            self._record(
                "Brak profilu psychologicznego użytkownika – odrzucam żądanie."
            )
            raise AdviceNotFoundError(
                "Brak profilu psychologicznego użytkownika – wykonaj najpierw test psychologiczny."
            )

        vocational_profile = await self._persona_provider.get_persona_by_type(
            user_id, "vocational"
        )

        normalized_message = request.user_message.strip()
        cache_key = (user_id, normalized_message)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self._record(
                "Cache hit: zwracam wcześniej wygenerowaną poradę dla tego użytkownika i wiadomości."
            )
            return cached_result

        # 1. Intent detection (kind)
        intent_match = await self._intent_detector.detect_preferred_kind(
            request.user_message
        )
        if intent_match:
            self._record(
                f"Rozpoznano prośbę o rodzaj: {intent_match.kind.value} (score={intent_match.score:.3f})"
            )
        else:
            self._record(
                "Brak jednoznacznej prośby o konkretny rodzaj porady (kind) – rozważam wszystkie rodzaje."
            )

        # 2. Build embedding input combining profiles and current message
        embedding_input_lines = [
            f"Profil psychologiczny użytkownika: {psych_profile}",
        ]
        if vocational_profile:
            embedding_input_lines.append(
                f"Profil zawodowy użytkownika: {vocational_profile}"
            )
        embedding_input_lines.append(
            f"Wiadomość użytkownika: {request.user_message}")
        embedding_input = "\n".join(embedding_input_lines)

        query_embedding = await self._embed_text(embedding_input)
        if not query_embedding:
            self._record("Nie udało się wygenerować embeddingu dla żądania.")
            raise AdviceNotFoundError(
                "Błąd podczas analizowania profilu użytkownika – spróbuj ponownie."
            )

        # 3. Fetch candidate advices filtered by intent (kind), if recognised
        if intent_match:
            candidates = await self._advice_repository.get_by_kind(intent_match.kind)
            if candidates:
                self._record(
                    f"Rozważam {len(candidates)} porad rodzaju {intent_match.kind.value}."
                )
            else:
                self._record(
                    f"Brak porad rodzaju {intent_match.kind.value} – rozważam pełny katalog."
                )
        else:
            candidates = ()

        if not candidates:
            candidates = await self._advice_repository.get_all()
            self._record(
                f"Rozpoznano łącznie {len(candidates)} porad w katalogu (bez filtrowania po kategoriach)."
            )

        # 4. Ensure advice embeddings exist (lazy cache + Supabase update)
        similarities: list[tuple[Advice, float]] = []
        for advice in candidates:
            if advice.id is None:
                # Should not happen for Supabase-backed repository, but be defensive
                self._record(
                    f"Porada '{advice.name}' nie ma identyfikatora – pomijam w trybie embeddingowym."
                )
                continue

            vector = await self._get_or_create_advice_embedding(advice)
            if not vector:
                self._record(
                    f"Nie udało się wygenerować embeddingu dla porady '{advice.name}' – pomijam."
                )
                continue
            score = OpenAIEmbeddingCategoryClassifier._cosine_similarity(
                query_embedding, vector
            )
            similarities.append((advice, score))

        if not similarities:
            self._record(
                "Żadna porada nie otrzymała poprawnego embeddingu – nie można nic zaproponować."
            )
            raise AdviceNotFoundError(
                "Brak porad możliwych do dopasowania w trybie embeddingowym."
            )

        # 5. Filter by minimal similarity threshold
        filtered = [
            (advice, score)
            for advice, score in similarities
            if score >= self._similarity_threshold
        ]
        if not filtered:
            self._record(
                f"Wszystkie porady miały zbyt niski wynik podobieństwa (< {self._similarity_threshold:.2f})."
            )
            raise AdviceNotFoundError(
                "Brak wystarczająco dopasowanej porady do profilu użytkownika."
            )

        # Ograniczamy się do TOP5 dopasowań, z mocnym naciskiem na TOP3
        filtered.sort(key=lambda item: item[1], reverse=True)
        top_candidates = filtered[:5]
        position_multipliers = [30.0, 7, 4, 0.5, 0.25]

        # 6. Probabilistic selection: similarity^2 z mnożnikami pozycji i intentu
        population: list[Advice] = []
        weights: list[float] = []
        for idx, (advice, score) in enumerate(top_candidates):
            weight = score * score
            weight *= (
                position_multipliers[idx]
                if idx < len(position_multipliers)
                else position_multipliers[-1]
            )
            if intent_match:
                if advice.kind == intent_match.kind:
                    weight *= 1.8
                else:
                    weight *= 0.15
            # Small jitter to avoid deterministic behaviour for ties
            weight *= random.uniform(0.9, 1.1)
            weights.append(max(weight, 0.01))
            population.append(advice)

        total_weight = sum(weights)
        if total_weight <= 0:
            self._record(
                "Suma wag w trybie embeddingowym wyniosła 0 – wybieram losowo spośród dopasowanych."
            )
            selected = random.choice([advice for advice, _ in top_candidates])
        else:
            selected = random.choices(
                population=population, weights=weights, k=1)[0]

        self._record(
            "Podsumowanie podobieństw embeddingowych (TOP 5):\n"
            + "\n".join(
                f"  - {advice.name} ({advice.kind.value}): score={score:.3f}"
                for advice, score in top_candidates
            )
        )
        self._record(
            f"Wybrana porada (embedding): {selected.name} ({selected.kind.value})."
        )

        chat_response = await self._response_generator.generate_response(
            advice=selected,
            request=request,
            categories=(),
            preferred_kind=intent_match.kind if intent_match else None,
        )
        recommendation = AdviceRecommendation(
            advice=selected, chat_response=chat_response)
        self._store_cached_result(cache_key, recommendation)
        return recommendation

    async def _get_or_create_advice_embedding(
        self, advice: Advice
    ) -> tuple[float, ...]:
        assert advice.id is not None
        if advice.id in self._embedding_cache:
            return self._embedding_cache[advice.id]

        # If Supabase already has an embedding, reuse it and cache locally
        if advice.embedding:
            vector = tuple(float(x) for x in advice.embedding)
            self._embedding_cache[advice.id] = vector
            return vector

        # Jeśli porada nie ma opisu w bazie, na razie ją pomijamy
        description = advice.description or ""
        if not description.strip():
            self._record(
                f"Porada '{advice.name}' nie ma opisu w bazie – pomijam w trybie embeddingowym."
            )
            return ()

        text = f"Rodzaj: {advice.kind.value}\n{description}"
        try:
            response = await self._client.embeddings.create(
                model=self._embeddings_model,
                input=[text],
            )
            embedding = tuple(response.data[0].embedding)
        except Exception as exc:  # pragma: no cover - network guard
            self._record(
                f"Błąd generowania embeddingu dla porady '{advice.name}': {exc}"
            )
            return ()

        # Persist in Supabase as cache
        try:
            await self._advice_repository.update_embedding(advice.id, embedding)
            self._record(
                f"Zapisano embedding w bazie dla porady '{advice.name}' (id={advice.id})."
            )
        except Exception as exc:  # pragma: no cover - defensive DB layer
            self._record(
                f"Nie udało się zapisać embeddingu w bazie dla porady '{advice.name}': {exc}"
            )

        self._embedding_cache[advice.id] = embedding
        return embedding

    async def _embed_text(self, text: str) -> tuple[float, ...]:
        try:
            response = await self._client.embeddings.create(
                model=self._embeddings_model,
                input=[text],
            )
        except Exception as exc:  # pragma: no cover - network guard
            self._record(f"Błąd generowania embeddingu zapytania: {exc}")
            return ()
        data = response.data or []
        if not data:
            return ()
        return tuple(data[0].embedding)

    def _get_cached_result(
        self, key: tuple[str, str]
    ) -> AdviceRecommendation | None:
        if self._cache_max_size <= 0:
            return None
        cached = self._result_cache.get(key)
        if cached is not None:
            self._result_cache.move_to_end(key)
        return cached

    def _store_cached_result(
        self, key: tuple[str, str], value: AdviceRecommendation
    ) -> None:
        if self._cache_max_size <= 0:
            return
        self._result_cache[key] = value
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self._cache_max_size:
            self._result_cache.popitem(last=False)


class EchoAdviceResponseGenerator(AdviceResponseGenerator):
    def set_log_sink(self, sink: Callable[[str], None]) -> None:  # pragma: no cover - compatibility hook
        pass

    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        selected_categories = ", ".join(
            categories) if categories else "general"
        preferred = preferred_kind.value if preferred_kind else "no specific kind"
        return (
            "Placeholder response: recommending "
            f"'{advice.name}' ({advice.kind.value}). "
            f"Categories matched: {selected_categories}. "
            f"Preferred kind: {preferred}."
        )


class NullAdviceIntentDetector(AdviceIntentDetector):
    async def detect_preferred_kind(
        self, user_message: str
    ) -> AdviceIntentMatch | None:
        return None


@dataclass(frozen=True)
class AdviceIntentDefinition:
    kind: AdviceKind
    description: str


class OpenAIEmbeddingAdviceIntentDetector(AdviceIntentDetector):
    def __init__(
        self,
        definitions: Sequence[AdviceIntentDefinition],
        *,
        settings: OpenAISettings | None = None,
        client: OpenAIClient | None = None,
        model: str | None = None,
        threshold: float = 0.4,
        log_limit: int = 3,
    ) -> None:
        if _RuntimeAsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install it with `pip install openai`."
            )
        if not definitions:
            raise ValueError("Advice intent detector requires definitions.")
        self._settings = settings or get_openai_settings()
        runtime_client: OpenAIClient = client or create_async_openai_client(
            self._settings
        )
        if not isinstance(runtime_client, _RuntimeAsyncOpenAI):
            raise TypeError(
                "client must be an instance of openai.AsyncOpenAI."
            )
        self._client = runtime_client
        self._model = model or self._settings.embeddings_model
        self._threshold = threshold
        self._definitions = tuple(definitions)
        self._log_limit = log_limit
        self._definition_embeddings: tuple[tuple[float, ...], ...] | None = None
        self._prepare_lock = asyncio.Lock()

    async def detect_preferred_kind(
        self, user_message: str
    ) -> AdviceIntentMatch | None:
        message = user_message.strip()
        if not message:
            return None
        await self._ensure_definition_embeddings()
        if not self._definition_embeddings:
            return None
        message_embedding = await self._embed_text(message)
        if not message_embedding:
            return None

        scored = []
        for vector, definition in zip(
            self._definition_embeddings, self._definitions
        ):
            score = OpenAIEmbeddingCategoryClassifier._cosine_similarity(
                message_embedding, vector
            )
            scored.append((score, definition))

        scored.sort(reverse=True, key=lambda item: item[0])
        if not scored:
            return None

        if self._log_limit and scored:
            lines = ["Rozpoznawanie intencji użytkownika (top wyniki):"]
            for score, definition in scored[: self._log_limit]:
                lines.append(
                    f"  - {definition.kind.value}: score={score:.3f}"
                )
            logger.info("\n".join(lines))

        best_score, best_definition = scored[0]
        if best_score < self._threshold:
            logger.info(
                "Najwyższy wynik dla rodzaju porady (%s) jest zbyt niski (%.3f < %.2f); traktujemy brak prośby.",
                best_definition.kind.value,
                best_score,
                self._threshold,
            )
            return None

        return AdviceIntentMatch(kind=best_definition.kind, score=best_score)

    async def _ensure_definition_embeddings(self) -> None:
        if self._definition_embeddings is not None:
            return
        async with self._prepare_lock:
            if self._definition_embeddings is not None:
                return
            inputs = [definition.description for definition in self._definitions]
            embeddings = await self._embed_texts(inputs)
            self._definition_embeddings = tuple(
                tuple(vec) for vec in embeddings)

    async def _embed_text(self, text: str) -> tuple[float, ...]:
        embeddings = await self._embed_texts([text])
        if not embeddings:
            return ()
        return tuple(embeddings[0])

    async def _embed_texts(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [tuple(item.embedding) for item in response.data]


class LLMAdviceResponseGenerator(AdviceResponseGenerator):
    def __init__(
        self,
        persona_provider: UserPersonaProvider,
        *,
        client: OpenAIClient | None = None,
        model: str | None = None,
    ) -> None:
        if _RuntimeAsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install it with `pip install openai`."
            )
        self._persona_provider = persona_provider
        runtime_client: OpenAIClient = client or create_async_openai_client(
            get_openai_settings())
        if not isinstance(runtime_client, _RuntimeAsyncOpenAI):
            raise TypeError(
                "client must be an instance of openai.AsyncOpenAI.")
        self._client = runtime_client
        self._model = model or os.getenv(
            "OPENAI_RESPONSE_MODEL") or "gpt-5-mini"
        self._reasoning_effort = os.getenv(
            "OPENAI_REASONING_EFFORT", "low") or "low"
        self._log_sink: Callable[[str], None] | None = None

    def set_log_sink(self, sink: Callable[[str], None]) -> None:
        self._log_sink = sink

    def _log(self, message: str) -> None:
        if self._log_sink:
            self._log_sink(message)
        else:
            logger.info(message)

    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        self._log(f"🚀 Rozpoczynam generowanie odpowiedzi LLM")
        self._log(f"📋 Porada: '{advice.name}' typu {advice.kind.value}")
        self._log(f"👤 User ID: {request.user_identifier.user_id}")
        self._log(f"💬 Wiadomość: '{request.user_message}'")
        self._log(
            f"🏷️ Kategorie: {', '.join(categories) if categories else 'brak'}")
        self._log(
            f"🎯 Preferowany rodzaj: {preferred_kind.value if preferred_kind else 'brak'}")
        persona_text: str | None = None
        user_id = request.user_identifier.user_id
        self._log(f"🔍 Szukam persony dla user_id: '{user_id}'")
        if user_id:
            try:
                persona_text = await self._persona_provider.get_persona(user_id)
                if persona_text:
                    self._log(f"✅ Pobrano personę: {persona_text[:50]}...")
                else:
                    self._log(
                        "⚠️ Brak zapisanego opisu osobowości – użyję neutralnego tonu")
            except Exception as e:
                self._log(f"❌ Błąd podczas pobierania persony: {e}")
                persona_text = None
        else:
            self._log(
                "⚠️ Brak identyfikatora użytkownika – persona nie będzie wykorzystana")

        advice_description = advice.description or "Brak dodatkowego opisu."
        categories_text = (
            ", ".join(
                categories) if categories else "brak kategorii dopasowanych wprost"
        )

        persona_prompt = (
            persona_text
            if persona_text
            else "Nie posiadamy szczegółowego opisu osobowości; odpowiedz w sposób serdeczny, empatyczny i uniwersalny."
        )

        system_prompt = (
            "Jesteś opiekuńczym asystentem."
            "Twoim zadaniem jest wygenerowanie odpowiedzi, która będzie dostosowana do osobowości użytkownika i będzie miała charakter zwięzły, wspierający i pełen nadziei. Pisz językiem naturalnym, bez meta-informacji takich jak \"Znam twój profil osobowości\" albo \"Nazwa porady to X\". Jeśli czasem porada jest słabo dopasowana, spróbuj to \"wyratować\" mówiąc trochę ogólnikami i lekko usprawiedliwiając wybór."
            "\nPisz 10 zdań, które:\n"
            "1. Odnoszą się do potrzeb użytkownika.\n"
            "2. Wyjaśniają, dlaczego wybrana porada może pomóc i jak ją zastosować.\n"
            "3. Utrzymują ton zwięzły, wspierający, opiekuńczy i pełen nadziei.\n"
            "4. Upewniają użytkownika, że jest zrozumiany.\n"
            "5. Całość ma być bardzo wyraźnie dostosowana do osobowości użytkownika, nawet wspomnij delikatnie o cechach osobowości użytkownika.\n"
        )

        author_line = f"Autor: {advice.author}\n" if advice.author else ""
        user_prompt = (
            "# Wiadomość użytkownika:\n"
            f"{request.user_message}\n\n"
            "# Porada dla użytkownika:\n"
            f"Nazwa: {advice.name}\n"
            f"Rodzaj: {advice.kind.value}\n"
            f"Opis: {advice_description}\n"
            f"{author_line}"
            "# Opis osobowości użytkownika:\n"
            f"{persona_prompt}\n\n"
        )

        self._log(f"📝 Utworzono system prompt ({len(system_prompt)} znaków)")
        self._log(f"📝 Utworzono user prompt ({len(user_prompt)} znaków)")

        try:
            self._log(f"🔄 Wysyłam zapytanie do OpenAI (model: {self._model})")
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            completion_text = response.choices[0].message.content
            if completion_text:
                completion_text = completion_text.strip()
                self._log(
                    f"✅ Odpowiedź LLM wygenerowana pomyślnie ({len(completion_text)} znaków)")
                return completion_text
            else:
                self._log("⚠️ LLM nie zwrócił treści – używam fallbacku")
        except Exception as exc:  # pragma: no cover - network guard
            self._log(f"❌ Błąd generowania odpowiedzi przez LLM: {exc}")
            self._log(f"📋 Typ błędu: {type(exc).__name__}")
            import traceback
            self._log(f"📋 Traceback: {traceback.format_exc()}")

        return self._fallback_response(advice, request.user_message, persona_text)

    def _fallback_response(
        self,
        advice: Advice,
        user_message: str,
        persona_text: str | None,
    ) -> str:
        raw_description = advice.description or (
            "Ta propozycja zawiera wartościowe wskazówki, które możesz spokojnie zastosować."
        )
        clean_description = re.sub(r"[.!?]+", "", raw_description).strip()
        persona_hint = (
            "Widzę, że Twój opis osobowości jest dla mnie ważny i chcę wesprzeć Cię w tym tonie."
            if persona_text
            else "Nie mam pełnego wglądu w Twoją osobowość, ale wciąż chcę wesprzeć Cię najlepiej jak potrafię."
        )
        persona_sentence = f"{persona_hint.rstrip('.!?')}."
        sentences = [
            "Dziękuję, że podzieliłeś się tym, co teraz przeżywasz.",
            persona_sentence,
            f"Wybrałam poradę '{advice.name}', ponieważ wierzę, że jest szczególnie adekwatna do Twojej sytuacji.",
            f"To {advice.kind.value.replace('_', ' ')} skupione na następującym kierunku działania: {clean_description}.",
            "Zachęcam Cię, abyś podszedł do tej propozycji krok po kroku i łagodnie wobec siebie.",
            "Zwróć uwagę na to, co w tej poradzie najbardziej rezonuje z Twoimi wartościami i aktualnymi potrzebami.",
            "Jeśli pojawią się trudniejsze emocje, nazwij je i pozwól sobie je odczuć zamiast je tłumić.",
            "Możesz zrobić pierwszy mały krok jeszcze dziś, nawet jeśli będzie symboliczny.",
            "Pamiętaj, że proszenie o wsparcie i korzystanie z narzędzi to oznaka odwagi, nie słabości.",
            "Jestem przy Tobie, by pomagać Ci przejść przez ten etap z troską i nadzieją.",
        ]
        return " ".join(sentences)


class MockAdviceResponseGenerator(AdviceResponseGenerator):
    """
    Extremely fast response generator for demo/testing purposes.
    Produces a deterministic ten-sentence reply without external calls.
    """

    def set_log_sink(self, sink: Callable[[str], None]) -> None:
        pass

    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        description = advice.description or "Ta propozycja ma potencjał wprowadzić pozytywną zmianę."
        categories_text = ", ".join(
            categories) if categories else "ogólne wskazówki"
        kind_text = advice.kind.value.replace("_", " ")
        preferred_text = preferred_kind.value if preferred_kind else "brak konkretnej prośby"
        sentences = [
            "Dziękuję za zaufanie i chęć rozmowy.",
            f"Wybrałam poradę '{advice.name}', ponieważ może odpowiadać na Twoje aktualne potrzeby.",
            f"To {kind_text}, które koncentruje się na następującym kierunku działania.",
            f"Opis porady brzmi: {description}",
            f"Kluczowe kategorie, które zadecydowały o wyborze, to: {categories_text}.",
            "Wdrażaj tę propozycję powoli, obserwując swoje emocje i reakcje ciała.",
            "Pozwól sobie na refleksję, które elementy porad najbardziej z Tobą rezonują.",
            f"Pamiętaj, że Twoja pierwotna prośba została odczytana jako: {preferred_text}.",
            "Nawet niewielki krok w stronę zmiany potrafi uruchomić pozytywną spiralę.",
            "Jestem przy Tobie, by dać Ci wsparcie, siłę i poczucie sprawczości.",
        ]
        return " ".join(sentences)


class StaticAdviceCategoryClassifier(AdviceCategoryClassifier):
    def __init__(self, fallback_category: str = "general") -> None:
        self._fallback_category = fallback_category

    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        if user_message.strip():
            return (
                CategoryMatch(
                    name=self._fallback_category,
                    score=1.0,
                    rank=1,
                ),
            )
        return ()


@dataclass(frozen=True)
class EmbeddingCategoryDefinition:
    name: str
    description: str


class OpenAIEmbeddingCategoryClassifier(AdviceCategoryClassifier):
    def __init__(
        self,
        categories: Sequence[EmbeddingCategoryDefinition],
        *,
        settings: OpenAISettings | None = None,
        client: OpenAIClient | None = None,
        model: str | None = None,
        similarity_threshold: float = 0.33,
        max_categories: int | None = 6,
    ) -> None:
        if _RuntimeAsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install it with `pip install openai`."
            )
        if not categories:
            raise ValueError(
                "OpenAIEmbeddingCategoryClassifier requires categories.")
        self._settings = settings or get_openai_settings()
        runtime_client: OpenAIClient = client or create_async_openai_client(
            self._settings
        )
        if not isinstance(runtime_client, _RuntimeAsyncOpenAI):
            raise TypeError(
                "client must be an instance of openai.AsyncOpenAI.")
        self._client = runtime_client
        self._model = model or self._settings.embeddings_model
        self._similarity_threshold = similarity_threshold
        self._max_categories = max_categories
        self._definitions = tuple(
            EmbeddingCategoryDefinition(
                name=definition.name.strip(),
                description=definition.description,
            )
            for definition in categories
        )
        self._category_embeddings: tuple[tuple[float, ...], ...] | None = None
        self._prepare_lock = asyncio.Lock()
        logger.info(
            "Initialized OpenAIEmbeddingCategoryClassifier with %d categories "
            "model=%s threshold=%.2f max_categories=%s",
            len(self._definitions),
            self._model,
            self._similarity_threshold,
            self._max_categories if self._max_categories is not None else "unlimited",
        )

    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        message = user_message.strip()
        if not message:
            return ()
        await self._ensure_category_embeddings()
        if not self._category_embeddings:
            return ()
        message_embedding = await self._embed_text(message)
        if not message_embedding:
            return ()

        scored_pairs: list[tuple[float, str]] = []
        for vector, definition in zip(self._category_embeddings, self._definitions):
            score = self._cosine_similarity(message_embedding, vector)
            if score >= self._similarity_threshold:
                scored_pairs.append((score, definition.name))

        scored_pairs.sort(reverse=True, key=lambda item: item[0])
        if scored_pairs:
            lines = [
                "Detected categories (similarity score):",
                *[
                    f"  - {slug}: {score:.3f}"
                    for score, slug in scored_pairs
                ],
            ]
            logger.info("\n".join(lines))
        else:
            logger.info(
                "No categories passed similarity threshold %.2f",
                self._similarity_threshold,
            )

        if self._max_categories is not None:
            scored_pairs = scored_pairs[: self._max_categories]

        matches = tuple(
            CategoryMatch(name=name, score=score, rank=index + 1)
            for index, (score, name) in enumerate(scored_pairs)
        )
        return matches

    async def _ensure_category_embeddings(self) -> None:
        if self._category_embeddings is not None:
            return
        async with self._prepare_lock:
            if self._category_embeddings is not None:
                return
            logger.info(
                "Embedding %d category descriptions via OpenAI model '%s'",
                len(self._definitions),
                self._model,
            )
            inputs = [definition.description for definition in self._definitions]
            embeddings = await self._embed_texts(inputs)
            self._category_embeddings = tuple(
                tuple(vector) for vector in embeddings)
            logger.info("Cached category embeddings.")

    async def _embed_text(self, text: str) -> tuple[float, ...]:
        embeddings = await self._embed_texts([text])
        if not embeddings:
            return ()
        return tuple(embeddings[0])

    async def _embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [tuple(item.embedding) for item in response.data]

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
