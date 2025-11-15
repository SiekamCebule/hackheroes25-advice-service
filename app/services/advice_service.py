from __future__ import annotations
import logging
import os
from typing import Protocol, Sequence

from app.integrations.openai import create_async_openai_client, get_openai_settings
from app.integrations.supabase import create_supabase_async_client
from app.repositories.advice_repository import SupabaseAdviceRepository, EmbeddingUpdatableAdviceRepository
from app.repositories.category_repository import SupabaseAdviceCategoryRepository
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
from app.repositories.mock_persona_repository import MockUserPersonaRepository
from app.services.advice_selection import (
    AdviceSelectionPipeline,
    AdviceIntentDefinition,
    OpenAIEmbeddingAdviceIntentDetector,
    EmbeddingCategoryDefinition,
    EchoAdviceResponseGenerator,
    LLMAdviceResponseGenerator,
    NullAdviceIntentDetector,
    OpenAIEmbeddingCategoryClassifier,
    StaticAdviceCategoryClassifier,
    PersonaEmbeddingAdviceSelectionPipeline,
)
from app.repositories.user_persona_repository import (
    NullUserPersonaProvider,
    SupabaseUserPersonaRepository,
    UserPersonaProvider,
)

logger = logging.getLogger(__name__)


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

    def get_latest_logs(self) -> Sequence[str]:
        getter = getattr(self._provider, "get_latest_events", None)
        if callable(getter):
            result = getter()
            if isinstance(result, Sequence) and all(isinstance(x, str) for x in result):
                return result
            return ()
        return ()


class SelectionEngine(Protocol):
    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        ...

    def get_latest_events(self) -> Sequence[str]:
        ...


class PipelineAdviceProvider:
    def __init__(self, pipeline: SelectionEngine) -> None:
        # We accept any selection engine that exposes `recommend` and
        # `get_latest_events` to keep this provider agnostic to the concrete strategy.
        self._pipeline = pipeline

    async def provide(self, request: AdviceRequestContext) -> AdviceRecommendation:
        return await self._pipeline.recommend(request)

    def get_latest_events(self) -> Sequence[str]:
        return self._pipeline.get_latest_events()


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
    intent_detector = NullAdviceIntentDetector()
    persona_repository = MockUserPersonaRepository(
        personas={
            "demo-ui-user": (
                "Introwertyczny umysł, potrzebujący przestrzeni, ceniący głębokie relacje, "
                "kochający spokój i analityczne podejście do codziennych wyzwań. "
                "W wolnym czasie skupia się na rozwoju osobistym poprzez czytanie i refleksję."
            )
        }
    )
    try:
        response_generator = LLMAdviceResponseGenerator(
            persona_provider=persona_repository,
            client=create_async_openai_client(get_openai_settings()),
            model=os.getenv("OPENAI_RESPONSE_MODEL") or "gpt-5-mini",
        )
    except RuntimeError:
        # Fallback to EchoAdviceResponseGenerator for demo purposes
        response_generator = EchoAdviceResponseGenerator()

    return AdviceSelectionPipeline(
        advice_repository=advice_repository,
        category_repository=category_repository,
        category_classifier=category_classifier,
        intent_detector=intent_detector,
        response_generator=response_generator,
    )


def build_supabase_advice_pipeline() -> SelectionEngine:
    client = create_supabase_async_client()
    advice_repository = EmbeddingUpdatableAdviceRepository(client)
    category_repository = SupabaseAdviceCategoryRepository(client)
    # Build category classifier only for the legacy, category-based mode.
    try:
        category_classifier = build_openai_category_classifier()
        logger.info(
            "Using OpenAIEmbeddingCategoryClassifier with %d category definitions.",
            len(_OPENAI_CATEGORY_DEFINITIONS),
        )
    except RuntimeError as openai_error:
        logger.warning(
            "Falling back to StaticAdviceCategoryClassifier due to OpenAI setup issue: %s",
            openai_error,
        )
        category_classifier = StaticAdviceCategoryClassifier()
    try:
        intent_detector = build_openai_intent_detector()
        logger.info(
            "Using EmbeddingAdviceIntentDetector with %d intent definitions.",
            len(_OPENAI_INTENT_DEFINITIONS),
        )
    except RuntimeError as openai_error:
        logger.warning(
            "Falling back to NullAdviceIntentDetector due to OpenAI setup issue: %s",
            openai_error,
        )
        intent_detector = NullAdviceIntentDetector()
    persona_provider = build_user_persona_provider(client)
    try:
        response_generator = build_openai_response_generator(persona_provider)
        logger.info(
            "Using LLMAdviceResponseGenerator with model '%s'.",
            os.getenv("OPENAI_RESPONSE_MODEL", "gpt-5-mini"),
        )
    except RuntimeError as openai_error:
        logger.warning(
            "Falling back to EchoAdviceResponseGenerator due to OpenAI setup issue: %s",
            openai_error,
        )
        response_generator = EchoAdviceResponseGenerator()

    selection_mode = os.getenv("ADVICE_SELECTION_MODE", "categories").lower()
    if selection_mode == "embedding":
        logger.info(
            "Using PersonaEmbeddingAdviceSelectionPipeline (mode=embedding, model=%s).",
            os.getenv("OPENAI_ADVICE_EMBEDDING_MODEL")
            or os.getenv("OPENAI_CATEGORY_MODEL")
            or "default embeddings model",
        )
        return PersonaEmbeddingAdviceSelectionPipeline(
            advice_repository=advice_repository,
            intent_detector=intent_detector,
            response_generator=response_generator,
            persona_provider=build_user_persona_provider(client),
            similarity_threshold=0.33,
            embeddings_model=os.getenv("OPENAI_ADVICE_EMBEDDING_MODEL"),
        )

    logger.info(
        "Using legacy category-based AdviceSelectionPipeline (mode=categories).")
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
    # Temporarily using default pipeline for chat UI demo (Supabase key invalid)
    # pipeline = build_default_advice_pipeline()
    pipeline = build_supabase_advice_pipeline()
    provider = PipelineAdviceProvider(pipeline=pipeline)
    return AdviceService(provider=provider)


def build_openai_category_classifier() -> OpenAIEmbeddingCategoryClassifier:
    if not _OPENAI_CATEGORY_DEFINITIONS:
        raise RuntimeError(
            "OPENAI category definitions are empty. Configure "
            "_OPENAI_CATEGORY_DEFINITIONS before building the classifier."
        )
    category_model = os.getenv("OPENAI_CATEGORY_MODEL")
    return OpenAIEmbeddingCategoryClassifier(
        categories=_OPENAI_CATEGORY_DEFINITIONS,
        similarity_threshold=0.3,
        max_categories=6,
        model=category_model,
    )


def build_openai_intent_detector() -> OpenAIEmbeddingAdviceIntentDetector:
    if not _OPENAI_INTENT_DEFINITIONS:
        raise RuntimeError(
            "OPENAI intent definitions are empty. Configure "
            "_OPENAI_INTENT_DEFINITIONS before building the detector."
        )
    intent_model = os.getenv("OPENAI_INTENT_MODEL")
    return OpenAIEmbeddingAdviceIntentDetector(
        definitions=_OPENAI_INTENT_DEFINITIONS,
        threshold=0.472,
        log_limit=5,
        model=intent_model,
    )


def build_user_persona_provider(client) -> UserPersonaProvider:
    table_name = os.getenv("SUPABASE_USER_PERSONA_TABLE")
    try:
        return SupabaseUserPersonaRepository(client, table_name=table_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Falling back to NullUserPersonaProvider due to issue with persona repository: %s",
            exc,
        )
        return NullUserPersonaProvider()


def build_openai_response_generator(
    persona_provider: UserPersonaProvider,
) -> LLMAdviceResponseGenerator:
    settings = get_openai_settings()
    client = create_async_openai_client(settings)
    model = os.getenv("OPENAI_RESPONSE_MODEL") or "gpt-5-mini"
    return LLMAdviceResponseGenerator(
        persona_provider=persona_provider,
        client=client,
        model=model,
    )


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
        categories=("mindfulness", "despair", "general"),
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

_OPENAI_CATEGORY_DEFINITIONS: Sequence[EmbeddingCategoryDefinition] = (
    EmbeddingCategoryDefinition(
        name="Poczucie winy",
        description="Użytkownik odczuwa winę, żal lub nadmierną samokrytykę lub szuka sposobu, by sobie wybaczyć.",
    ),
    EmbeddingCategoryDefinition(
        name="Bezsilność",
        description="Użytkownik ma poczucie braku wpływu na sytuację i potrzebuje wsparcia w odzyskaniu sprawczości.",
    ),
    EmbeddingCategoryDefinition(
        name="Depresja",
        description="Użytkownik przeżywa depresję lub głęboką utratę nadzieji i sensu; lub interesuje się tematem depresji;",
    ),
    EmbeddingCategoryDefinition(
        name="Wstyd",
        description="Użytkownik doświadcza wstydu, unika oceny lub ma trudność z akceptacją siebie.",
    ),
    EmbeddingCategoryDefinition(
        name="Rozpacz",
        description="Użytkownik przeżywa silny smutek, utratę nadziei lub głęboki kryzys emocjonalny.",
    ),
    EmbeddingCategoryDefinition(
        name="Bardzo silny kryzys psychiczny",
        description="Użytkownik sygnalizuje intensywne cierpienie psychiczne i potrzebuje natychmiastowego wsparcia.",
    ),
    EmbeddingCategoryDefinition(
        name="Strach",
        description="Użytkownik opisuje lęk, poczucie zagrożenia lub niepokój związany z przyszłością.",
    ),
    EmbeddingCategoryDefinition(
        name="Lęk społeczny",
        description="Użytkownik boi się kontaktu z ludźmi, odrzucenia lub oceny społecznej.",
    ),
    EmbeddingCategoryDefinition(
        name="Uzależnienie",
        description="Użytkownik zmaga się z uzależnieniem lub zachowaniami kompulsywnymi.",
    ),
    EmbeddingCategoryDefinition(
        name="Złość",
        description="Użytkownik doświadcza silnej złości, frustracji lub agresji i szuka sposobów jej regulacji.",
    ),
    EmbeddingCategoryDefinition(
        name="Pogarda",
        description="Użytkownik wyraża pogardę wobec siebie lub innych, często jako mechanizm obronny.",
    ),
    EmbeddingCategoryDefinition(
        name="Odwaga",
        description="Użytkownik szuka siły do działania mimo lęku, potrzebuje wsparcia w podejmowaniu decyzji.",
    ),
    EmbeddingCategoryDefinition(
        name="Akceptacja",
        description="Użytkownik pracuje nad przyjęciem siebie, emocji lub trudnych sytuacji życiowych. Może też mieć problem z akceptacją lub przepracowaniem emocji.",
    ),
    EmbeddingCategoryDefinition(
        name="Miłość",
        description="Użytkownik mówi o potrzebie miłości, bliskości lub problemach w relacjach uczuciowych.",
    ),
    EmbeddingCategoryDefinition(
        name="Relacje międzyludzkie",
        description="Użytkownik analizuje więzi z innymi ludźmi, konflikty lub samotność.",
    ),
    EmbeddingCategoryDefinition(
        name="Motywacja",
        description="Użytkownik szuka inspiracji, energii do działania lub przezwyciężenia apatii.",
    ),
    EmbeddingCategoryDefinition(
        name="Rozwój osobisty",
        description="Użytkownik dąży do samodoskonalenia i lepszego zrozumienia siebie.",
    ),
    EmbeddingCategoryDefinition(
        name="Uczenie się",
        description="Użytkownik interesuje się procesem nauki, koncentracją lub efektywnością poznawczą.",
    ),
    EmbeddingCategoryDefinition(
        name="Zazdrość",
        description="Użytkownik doświadcza porównywania się z innymi, poczucia braku lub zazdrości.",
    ),
    EmbeddingCategoryDefinition(
        name="Planowanie",
        description="Użytkownik chce uporządkować działania, wyznaczyć cele lub poprawić organizację.",
    ),
    EmbeddingCategoryDefinition(
        name="Nawyki",
        description="Użytkownik chce dowiedzieć się czegoś o nawykach, ma destrukcyjne nawyki lub pracuje nad zmianą codziennych zachowań lub wprowadzeniem rutyny.",
    ),
    EmbeddingCategoryDefinition(
        name="Psychologia",
        description="Użytkownik porusza tematy psychologiczne, emocjonalne lub poznawcze, a szczególnie szuka wsparcia psychologicznego lub psychoterapeutycznego.",
    ),
    EmbeddingCategoryDefinition(
        name="Zdrowie",
        description="Użytkownik skupia się na kondycji fizycznej, psychicznej lub równowadze ciała i umysłu.",
    ),
    EmbeddingCategoryDefinition(
        name="Aktywność fizyczna",
        description="Użytkownik interesuje się ruchem, sportem lub wpływem aktywności na psychikę.",
    ),
    EmbeddingCategoryDefinition(
        name="Relaksacja",
        description="Użytkownik szuka sposobów na odprężenie i redukcję stresu.",
    ),
    EmbeddingCategoryDefinition(
        name="Inspiracja",
        description="Użytkownik poszukuje inspirujących treści lub odzyskania poczucia sensu.",
    ),
    EmbeddingCategoryDefinition(
        name="Sukces",
        description="Użytkownik rozważa swoje osiągnięcia, cele lub presję związaną z sukcesem.",
    ),
    EmbeddingCategoryDefinition(
        name="Marzenia",
        description="Użytkownik mówi o swoich pragnieniach, wizjach przyszłości lub potrzebie celu.",
    ),
    EmbeddingCategoryDefinition(
        name="Szczęście",
        description="Użytkownik szuka równowagi emocjonalnej, spokoju lub spełnienia, lub czuje potrzebę szczęścia w życiu.",
    ),
    EmbeddingCategoryDefinition(
        name="Przebodźcowanie",
        description="Użytkownik czuje się przytłoczony nadmiarem informacji lub bodźców.",
    ),
    EmbeddingCategoryDefinition(
        name="Sprawy egzystencjalne",
        description="Użytkownik rozmyśla o sensie życia, przemijaniu lub tożsamości, a także innych głębokich sprawach życiowych.",
    ),
    EmbeddingCategoryDefinition(
        name="Media społecznościowe",
        description="Użytkownik porusza temat wpływu mediów społecznościowych na psychikę lub relacje.",
    ),
    EmbeddingCategoryDefinition(
        name="Ekspresja emocji",
        description="Użytkownik chce lepiej wyrażać swoje emocje lub mówi o problemach z wyrażaniem emocji lub o stłumionych emocjach.",
    ),
    EmbeddingCategoryDefinition(
        name="Umysł",
        description="Użytkownik interesuje się mechanizmami myślenia, świadomości lub introspekcji.",
    ),
    EmbeddingCategoryDefinition(
        name="Biznes",
        description="Użytkownik myśli o biznesie, przedsiębiorczości lub o pracy w firmie.",
    ),
    EmbeddingCategoryDefinition(
        name="Rodzina",
        description="Użytkownik myśli o rodzinie, relacjach rodzinnych, w tym traumach i problemach rodzinnych.",
    ),
    EmbeddingCategoryDefinition(
        name="Duchowość",
        description="Użytkownik myśli o duchowości, religii, mistycyzmie lub innych sprawach duchowych.",
    ),
    EmbeddingCategoryDefinition(
        name="Trauma",
        description="Użytkownik myśli o traumie, cierpieniu psychicznym lub fizycznym, lub próbuje przezwyciężyć traumę.",
    ),
    EmbeddingCategoryDefinition(
        name="Komunikacja",
        description="Użytkownik myśli o komunikacji, dialogu, komunikacji z innymi osobami lub z samym sobą.",
    ),
    EmbeddingCategoryDefinition(
        name="Public speaking",
        description="Użytkownik myśli o publicznym wystąpieniu, prelekcji, wykładzie lub innych formach komunikacji publicznej.",
    ),
)

_OPENAI_INTENT_DEFINITIONS: Sequence[AdviceIntentDefinition] = (
    AdviceIntentDefinition(
        kind=AdviceKind.BOOK,
        description="Użytkownik JAWNIE prosi o książkę lub rekomendację do przeczytania.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.MOVIE,
        description="Użytkownik JAWNIE szuka filmu lub serialu do obejrzenia.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.MUSIC,
        description="Użytkownik JAWNIE prosi o muzykę, utwór lub playlistę.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.YOUTUBE_VIDEO,
        description="Użytkownik JAWNIE chce otrzymać link lub propozycję filmiku na YouTube (w skrócie YT).",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.ARTICLE,
        description="Użytkownik JAWNIE szuka artykułu, publikacji lub wpisu do przeczytania.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.HABIT,
        description="Użytkownik JAWNIE prosi o nawyk, rutynę lub małe zadanie do wdrożenia.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.ADVICE,
        description="Użytkownik JAWNIE prosi ogólnie o poradę lub wskazówkę, nie precyzując formatu.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.CONCEPT,
        description="Użytkownik JAWNIE prosi o wyjaśnienie pojęcia, idei albo definicji.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.PSYCHOTHERAPY,
        description="Użytkownik JAWNIE prosi o konkretną terapię lub poszukuje materiałów terapeutycznych.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.PODCAST,
        description="Użytkownik JAWNIE prosi o podcast lub odcinek audio do posłuchania.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.QUOTE,
        description="Użytkownik JAWNIE prosi o cytat, sentencję lub inspirujące słowa, ale nie chodzi mu o konkretną osobę.",
    ),
    AdviceIntentDefinition(
        kind=AdviceKind.PERSON,
        description="Użytkownik prosi o informacje o konkretnej osobie, jej życiu, działalności lub wpływie. Np. użytkownik mówi \"znajdź osobę\" lub coś w tym stylu. Albo np. \"się zainspirować człowiekiem sukcesu\"",
    ),
)
