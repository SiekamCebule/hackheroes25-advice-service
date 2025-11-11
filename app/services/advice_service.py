from __future__ import annotations

from typing import Protocol, Sequence

from app.models.advice import (
    Advice,
    AdviceKind,
    AdviceRecommendation,
    AdviceRequestContext,
    AdviceResponsePayload,
)
from app.repositories.advice_repository import AdviceRepository, InMemoryAdviceRepository
from app.repositories.category_repository import (
    AdviceCategoryRepository,
    StaticAdviceCategoryRepository,
)
from app.services.advice_selection import (
    AdviceSelectionPipeline,
    EchoAdviceResponseGenerator,
    StaticAdviceCategoryClassifier,
    TrivialAdviceIntentDetector,
)


class AdviceProvider(Protocol):
    async def provide(self, request: AdviceRequestContext) -> AdviceRecommendation:
        raise NotImplementedError


class AdviceService:
    def __init__(self, provider: AdviceProvider):
        self._provider = provider

    async def get_advice(self, request: AdviceRequestContext) -> AdviceRecommendation:
        return await self._provider.provide(request)

    async def get_advice_response(
        self, request: AdviceRequestContext
    ) -> AdviceResponsePayload:
        recommendation = await self.get_advice(request)
        return AdviceResponsePayload.from_recommendation(recommendation)


class PipelineAdviceProvider:
    def __init__(self, pipeline: AdviceSelectionPipeline) -> None:
        self._pipeline = pipeline

    async def provide(self, request: AdviceRequestContext) -> AdviceRecommendation:
        return await self._pipeline.recommend(request)


def build_default_advice_repository() -> AdviceRepository:
    return InMemoryAdviceRepository(
        advice_items=_DEFAULT_ADVICE_ITEMS,
    )


def build_default_category_repository() -> AdviceCategoryRepository:
    return StaticAdviceCategoryRepository(categories=_DEFAULT_CATEGORIES)


def build_default_advice_pipeline() -> AdviceSelectionPipeline:
    advice_repository = build_default_advice_repository()
    category_repository = build_default_category_repository()
    category_classifier = StaticAdviceCategoryClassifier()
    intent_detector = TrivialAdviceIntentDetector()
    response_generator = EchoAdviceResponseGenerator()

    return AdviceSelectionPipeline(
        advice_repository=advice_repository,
        category_repository=category_repository,
        category_classifier=category_classifier,
        intent_detector=intent_detector,
        response_generator=response_generator,
    )


def get_advice_service() -> AdviceService:
    # TODO: replace build_default_advice_pipeline with production-ready dependencies
    # (Supabase repositories, embedding-based classifiers, and LLM-powered response generators).
    pipeline = build_default_advice_pipeline()
    provider = PipelineAdviceProvider(pipeline=pipeline)
    return AdviceService(provider=provider)


_DEFAULT_CATEGORIES: Sequence[str] = (
    "general",
    "sadness",
    "longing",
    "despair",
    "habit",
    "motivation",
    "mindfulness",
    "psychotherapy",
    "calm",
    "anxiety",
)

_DEFAULT_ADVICE_ITEMS: Sequence[Advice] = (
    Advice(
        name="Atomic Habits",
        kind=AdviceKind.BOOK,
        description="A practical guide to building good habits and breaking bad ones.",
        link_url="https://example.com/atomic-habits",
        image_url="https://example.com/atomic-habits.jpg",
        author="James Clear",
        categories=("habit", "motivation"),
    ),
    Advice(
        name="Lo-Fi Focus",
        kind=AdviceKind.MUSIC,
        description="A mellow playlist to help you regain focus and calm.",
        link_url="https://example.com/lofi-focus",
        image_url="https://example.com/lofi-focus.jpg",
        categories=("mindfulness", "calm"),
    ),
    Advice(
        name="Guided Breathing Exercise",
        kind=AdviceKind.ADVICE,
        description="A short breathing routine for moments of overwhelm.",
        link_url="https://example.com/breathing-exercise",
        categories=("mindfulness", "despair"),
    ),
    Advice(
        name="Meaningful Quote Collection",
        kind=AdviceKind.QUOTE,
        description="Curated quotes to bring comfort when you feel alone.",
        link_url="https://example.com/meaningful-quotes",
        categories=("sadness", "longing"),
    ),
    Advice(
        name="First Psychotherapy Session Guide",
        kind=AdviceKind.PSYCHOTHERAPY,
        description="What to expect and how to prepare for your first visit.",
        link_url="https://example.com/therapy-guide",
        categories=("psychotherapy", "anxiety"),
    ),
)
