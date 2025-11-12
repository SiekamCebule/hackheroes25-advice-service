from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import random
import unicodedata
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from app.models.advice import Advice, AdviceKind, AdviceRecommendation, AdviceRequestContext
from app.repositories.advice_repository import AdviceRepository
from app.repositories.category_repository import AdviceCategoryRepository
from app.integrations.openai import (
    OpenAISettings,
    create_async_openai_client,
    get_openai_settings,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as OpenAIClient  # type: ignore[import]
else:
    OpenAIClient = Any  # type: ignore[misc]

try:  # pragma: no cover - runtime availability check
    from openai import AsyncOpenAI as _RuntimeAsyncOpenAI
except ImportError:
    _RuntimeAsyncOpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class AdviceCategoryClassifier(Protocol):
    async def infer_categories(self, user_message: str) -> Sequence[str]:
        raise NotImplementedError


class AdviceIntentDetector(Protocol):
    async def detect_preferred_kind(self, user_message: str) -> AdviceKind | None:
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
    ) -> None:
        self._advice_repository = advice_repository
        self._category_repository = category_repository
        self._category_classifier = category_classifier
        self._intent_detector = intent_detector
        self._response_generator = response_generator

    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        logger.info(
            "Pipeline recommend start user=%s message_len=%d",
            request.user_identifier.user_id or "anonymous",
            len(request.user_message),
        )
        categories = await self._infer_categories(request.user_message)
        logger.info("Pipeline categories=%s", categories or "[]")
        preferred_kind = await self._intent_detector.detect_preferred_kind(
            request.user_message
        )
        logger.info("Pipeline preferred_kind=%s", preferred_kind or "none")

        advice = await self._select_advice(preferred_kind, categories)
        if advice is None:
            raise AdviceNotFoundError(
                "No advice found for the given criteria.")

        chat_response = await self._response_generator.generate_response(
            advice=advice,
            request=request,
            categories=categories,
            preferred_kind=preferred_kind,
        )
        return AdviceRecommendation(advice=advice, chat_response=chat_response)

    async def _infer_categories(self, user_message: str) -> Sequence[str]:
        inferred = await self._category_classifier.infer_categories(user_message)
        if not inferred:
            logger.info("Classifier produced no categories for message.")
            return ()
        known_categories = await self._category_repository.get_all()
        variant_map = self._build_known_category_map(known_categories)
        unique_categories = []
        for category in inferred:
            normalized = category.lower()
            matched_name = self._match_category_to_known(
                normalized, variant_map)
            if matched_name:
                canonical = matched_name
                if canonical not in unique_categories:
                    unique_categories.append(canonical)
                    logger.info(
                        "Classifier category '%s' matched repository name '%s'.",
                        normalized,
                        canonical,
                    )
            else:
                logger.info(
                    "Classifier category '%s' not in known set. Variants tried=%s",
                    normalized,
                    self._build_category_variants(normalized),
                )
        if not unique_categories:
            logger.info(
                "No classifier categories matched known repository categories.")
        return tuple(unique_categories)

    async def _select_advice(
        self,
        preferred_kind: AdviceKind | None,
        categories: Sequence[str],
    ) -> Advice | None:
        candidates: Sequence[Advice] = ()

        if preferred_kind is not None:
            if categories:
                candidates = await self._advice_repository.get_by_kind_and_containing_any_category(
                    preferred_kind,
                    categories,
                )
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)
                logger.info(
                    "Candidates from kind filtering count=%d", len(candidates)
                )

        if not candidates:
            if categories:
                all_advice = await self._advice_repository.get_all()
                candidates = tuple(
                    advice
                    for advice in all_advice
                    if self._contains_any_category(advice, categories)
                )
                logger.info(
                    "Candidates after category scan count=%d", len(candidates)
                )

        if not candidates:
            candidates = await self._advice_repository.get_all()
            logger.info("Candidates fallback to all count=%d", len(candidates))

        selected = self._rank_candidates(candidates, categories)
        if selected:
            logger.info("Pipeline selected advice name='%s'", selected.name)
        else:
            logger.warning(
                "Pipeline ranking returned None despite candidates.")
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
        self, candidates: Sequence[Advice], categories: Sequence[str]
    ) -> Advice | None:
        if not candidates:
            return None

        if not categories:
            return random.choice(tuple(candidates))

        category_set = {category.lower() for category in categories}

        scored_candidates: list[tuple[int, Advice]] = []
        for candidate in candidates:
            candidate_categories = {
                category.lower() for category in candidate.categories
            }
            overlap = len(candidate_categories & category_set)
            scored_candidates.append((overlap, candidate))

        grouped: dict[int, list[Advice]] = {}
        max_overlap = 0
        for score, candidate in scored_candidates:
            grouped.setdefault(score, []).append(candidate)
            if score > max_overlap:
                max_overlap = score

        if max_overlap == 0:
            return None

        max_considered = min(max_overlap, 6)
        for target_score in range(max_considered, 0, -1):
            candidates_at_score = grouped.get(target_score)
            if candidates_at_score:
                logger.info(
                    "Ranking selecting from %d candidates with overlap=%d",
                    len(candidates_at_score),
                    target_score,
                )
                return random.choice(candidates_at_score)

        return None

    @staticmethod
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


class EchoAdviceResponseGenerator(AdviceResponseGenerator):
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


class TrivialAdviceIntentDetector(AdviceIntentDetector):
    async def detect_preferred_kind(self, user_message: str) -> AdviceKind | None:
        return None


class StaticAdviceCategoryClassifier(AdviceCategoryClassifier):
    def __init__(self, fallback_category: str = "general") -> None:
        self._fallback_category = fallback_category

    async def infer_categories(self, user_message: str) -> Sequence[str]:
        if user_message.strip():
            return (self._fallback_category,)
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
        similarity_threshold: float = 0.25,
        max_categories: int | None = 5,
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
                name=definition.name.lower(), description=definition.description
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

    async def infer_categories(self, user_message: str) -> Sequence[str]:
        message = user_message.strip()
        if not message:
            return ()
        await self._ensure_category_embeddings()
        if not self._category_embeddings:
            return ()
        message_embedding = await self._embed_text(message)
        if not message_embedding:
            return ()

        scored: list[tuple[float, str]] = []
        for vector, definition in zip(self._category_embeddings, self._definitions):
            score = self._cosine_similarity(message_embedding, vector)
            if score >= self._similarity_threshold:
                scored.append((score, definition.name))

        scored.sort(reverse=True, key=lambda item: item[0])
        if scored:
            lines = [
                "Detected categories (similarity score):",
                *[
                    f"  - {slug}: {score:.3f}"
                    for score, slug in scored
                ],
            ]
            logger.info("\n".join(lines))
        else:
            logger.info(
                "No categories passed similarity threshold %.2f",
                self._similarity_threshold,
            )

        if self._max_categories is not None:
            scored = scored[: self._max_categories]
        return tuple(slug for _, slug in scored)

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
