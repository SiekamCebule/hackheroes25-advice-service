from __future__ import annotations

import asyncio
import math
import re
from collections import defaultdict
from typing import Mapping, Sequence
import os

from app.integrations.openai import create_async_openai_client, get_openai_settings
from app.integrations.supabase import create_supabase_async_client
from app.models.tests import (
    PSYCHO_TRAITS,
    VOCATION_TRAITS,
    PsychologyTestRequest,
    TestSubmissionResponse,
    TraitImpact,
    VocationalTestRequest,
)
from app.repositories.test_repository import TestRepository
from app.repositories.user_persona_repository import (
    NullUserPersonaProvider,
    SupabaseUserPersonaRepository,
    UserPersonaProvider,
)

# --- Question configuration ---

PSYCHO_QUESTION_IMPACTS: Sequence[Sequence[TraitImpact]] = (
    (TraitImpact("ekstrawersja", 1.0),),  # 1
    (TraitImpact("ekstrawersja", 1.0),),  # 2
    (TraitImpact("sumienność", 1.0),),  # 3
    (TraitImpact("sumienność", 1.0, reverse=True),),  # 4
    (TraitImpact("ugodowość", 1.0),),  # 5
    (TraitImpact("ugodowość", 1.0, reverse=True),),  # 6
    (TraitImpact("stabilność_emocjonalna", 1.0, reverse=True),),  # 7
    (TraitImpact("stabilność_emocjonalna", 1.0),),  # 8
    (
        TraitImpact("stabilność_emocjonalna", 0.6),
        TraitImpact("koncentracja", 0.4),
    ),  # 9
    (TraitImpact("kreatywność", 1.0),),  # 10
    (TraitImpact("kreatywność", 1.0),),  # 11
    (TraitImpact("myślenie_logiczne", 1.0),),  # 12
    (TraitImpact("myślenie_logiczne", 1.0, reverse=True),),  # 13
    (TraitImpact("koncentracja", 1.0, reverse=True),),  # 14
    (TraitImpact("koncentracja", 1.0, reverse=True),),  # 15
    (TraitImpact("przywództwo", 1.0),),  # 16
    (TraitImpact("przywództwo", 1.0),),  # 17
)

VOCATION_QUESTION_IMPACTS: Sequence[Sequence[TraitImpact]] = (
    # 1. Dobrze odnajduję się w pracy grupowej
    (TraitImpact("praca_z_ludzmi", 1.0),),
    # 2. Efektywniej pracuję, jeśli jestem sam
    (TraitImpact("praca_z_ludzmi", 1.0, reverse=True),),
    # 3. Lubię wykonywać drobne naprawy w domu, albo inną pracę manualną
    (TraitImpact("majstrowanie", 1.0),),
    # 4. Lubię/lubiałem pisać rozprawki w szkole, albo prace naukowe na studiach
    (TraitImpact("pisanie", 1.0),),
    # 5. Jestem zaawansowany w ogólnej obsłudze komputera
    (TraitImpact("obsluga_komputera", 1.0),),
    # 6. Mam wyczucie estetyki
    (TraitImpact("sztuka_design", 1.0),),
    # 7. Lubię wyrażać swoją opinię na dany temat
    (TraitImpact("pisanie", 1.0),),
    # 8. Regularnie obcuję ze sztuką (teatr, opera, wystawy i inne)
    (TraitImpact("sztuka_design", 1.0),),
    # 9. Umiem myśleć algorytmicznie
    (TraitImpact("programowanie", 1.0),),
    # 10. Lubię rozwiązywać złożone problemy
    (TraitImpact("programowanie", 1.0), TraitImpact("analiza_danych", 0.7),),
    # 11. Chcę tworzyć coś, z czego będą korzystać inni ludzie
    (TraitImpact("programowanie", 0.8), TraitImpact("zarzadzanie_projektem", 0.6),),
    # 12. Szybko uczę się języków obcych
    (TraitImpact("jezyki_obce", 1.0),),
    # 13. Jestem sprawny fizycznie
    (TraitImpact("praca_terenowa", 1.0),),
    # 14. Cenię sobie kontakt z naturą
    (TraitImpact("kontakt_z_natura", 1.0),),
    # 15. Poszukuję pracy która zapewni mi stabilność finansową i bezpieczeństwo
    (TraitImpact("priorytet_stabilnosc", 1.0),),
    # 16. Wolę pracować zdalnie, z dowolnego miejsca na świecie
    (TraitImpact("praca_zdalna", 1.0),),
    # 17. Jestem kiepski w wystąpieniach publicznych i unikam ich jak ognia
    (TraitImpact("wystapienia_publiczne", 1.0, reverse=True),),
    # 18. Bardzo cenię sobie aspekt finansowy pracy
    (TraitImpact("priorytet_pieniadze", 1.0),),
    # 19. Chcę by praca rozwijała mnie jako człowieka, nawet kosztem mniejszych zarobków lub mniejszej stabilności
    (TraitImpact("priorytet_rozwoj", 1.0),),
    # 20. Dobrze wyciągam wnioski na podstawie danych
    (TraitImpact("analiza_danych", 1.0),),
    # 21. Interesuję się anatomią człowieka i pracą lekarzy
    (TraitImpact("biologia_medycyna", 1.0),),
    # 22. Mam dobrą pamięć do pojęć
    (TraitImpact("biologia_medycyna", 0.8), TraitImpact("analiza_danych", 0.6),),
    # 23. Lubię brać udział w zorganizowanych projektach grupowych, w szkole, na uczelni lub w pracy
    (TraitImpact("praca_z_ludzmi", 1.0), TraitImpact("zarzadzanie_projektem", 0.7),),
)


PSYCHO_TRAIT_DESCRIPTIONS: Mapping[str, str] = {
    "ekstrawersja": "Potrzeba interakcji, energii społecznej i bycia w centrum wydarzeń.",
    "ugodowość": "Empatia, współpraca i szukanie harmonii w relacjach.",
    "sumienność": "Organizacja, konsekwencja, planowanie i dotrzymywanie terminów.",
    "stabilność_emocjonalna": "Łatwość regulowania emocji, spokój i odporność na stres.",
    "kreatywność": "Skłonność do nieszablonowych rozwiązań i eksperymentowania.",
    "myślenie_logiczne": "Preferencja dla analizowania danych i decyzji opartych na faktach.",
    "koncentracja": "Umiejętność skupienia się na jednej aktywności i domykania zadań.",
    "przywództwo": "Naturalna rola lidera, wpływ na grupę, chęć podejmowania decyzji.",
}

VOCATION_TRAIT_DESCRIPTIONS: Mapping[str, str] = {
    "majstrowanie": "Skłonność do pracy manualnej i konstruowania rzeczy.",
    "kontakt_z_natura": "Potrzeba obcowania z naturą i prac terenowych.",
    "obsluga_komputera": "Sprawność w obsłudze narzędzi cyfrowych.",
    "zarzadzanie_projektem": "Planowanie, przydzial zadań i pilnowanie realizacji.",
    "programowanie": "Pisanie kodu, budowanie aplikacji, rozwiązywanie problemów technicznych.",
    "biologia_medycyna": "Zainteresowanie zdrowiem, biologią i troską o innych.",
    "sztuka_design": "Ekspresja artystyczna, estetyka i projektowanie.",
    "analiza_danych": "Praca z liczbami, raportami, wnioskami.",
    "jezyki_obce": "Łatwość w nauce i stosowaniu języków obcych.",
    "praca_terenowa": "Aktywność fizyczna, praca w ruchu poza biurem.",
    "praca_zdalna": "Preferencja do pracy z domu, elastyczność miejsca.",
    "wystapienia_publiczne": "Komunikacja przed publicznością, prezentacje.",
    "pisanie": "Tworzenie tekstów, storytelling, copywriting.",
    "priorytet_pieniadze": "Motywacja finansowa.",
    "priorytet_rozwoj": "Nacisk na rozwój osobisty.",
    "priorytet_stabilnosc": "Poszukiwanie bezpieczeństwa i przewidywalności.",
    "praca_z_ludzmi": "Chęć pracy zespołowej zamiast solo.",
}


PSYCHO_OPEN_QUESTION_PROMPTS: Sequence[str] = (
    "Pytanie 18 (psychologia) – Czego brakuje Ci do pełni szczęścia?",
    "Pytanie 19 (psychologia) – Jaki byłby Twój wymarzony partner życiowy?",
    "Pytanie 20 (psychologia) – Jak wyglądałby Twój wymarzony dzień?",
    "Pytanie 21 (psychologia) – Czego w życiu najbardziej się boisz?",
)

VOCATION_OPEN_QUESTION_PROMPTS: Sequence[str] = (
    "Pytanie 24 (zawodowy) – Opisz sytuację z pracy, która dała Ci największą satysfakcję i dlaczego.",
    "Pytanie 25 (zawodowy) – Jakie umiejętności lub wiedzę chcesz najbardziej rozwijać w karierze?",
    "Pytanie 26 (zawodowy) – Czego najbardziej unikasz lub czego się obawiasz w kontekście pracy zawodowej?",
    "Pytanie 27 (zawodowy) – Jak praca wpływa na Twoje życie prywatne i jaki balans jest dla Ciebie idealny?",
)


def _normalize_scores(raw_scores: Mapping[str, float], max_score: Mapping[str, float]) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for trait, value in raw_scores.items():
        cap = max_score.get(trait, 1.0) or 1.0
        normalized[trait] = min(max(value / cap, 0.0), 1.0)
    return normalized


class OpenAnswerTraitClassifier:
    def __init__(
        self,
        trait_descriptions: Mapping[str, str],
        *,
        threshold: float = 0.45,
        max_boost: float = 0.2,
        model: str | None = None,
        allow_negative: bool = True,
        negative_weight: float = 1.0,
    ) -> None:
        self._trait_descriptions = trait_descriptions
        settings = get_openai_settings()
        self._client = create_async_openai_client(settings)
        self._model = model or settings.embeddings_model
        self._threshold = threshold
        self._max_boost = max_boost
        self._trait_embeddings: list[tuple[str, Sequence[float]]] | None = None
        self._prepare_lock = asyncio.Lock()
        self._negative_trait_embeddings: list[tuple[str,
                                                    Sequence[float]]] | None = None
        self._allow_negative = allow_negative
        self._negative_weight = negative_weight

    async def _ensure_embeddings(self) -> None:
        if (
            self._trait_embeddings is not None
            and (not self._allow_negative or self._negative_trait_embeddings is not None)
        ):
            return
        async with self._prepare_lock:
            if (
                self._trait_embeddings is not None
                and (not self._allow_negative or self._negative_trait_embeddings is not None)
            ):
                return
            items = list(self._trait_descriptions.items())
            texts = [desc for _, desc in items]
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            embeddings = [tuple(item.embedding) for item in response.data]
            self._trait_embeddings = list(
                zip(self._trait_descriptions.keys(), embeddings)
            )
            if self._allow_negative:
                negative_texts = [
                    self._build_negative_description(trait, desc) for trait, desc in items
                ]
                negative_response = await self._client.embeddings.create(
                    model=self._model,
                    input=negative_texts,
                )
                negative_embeddings = [
                    tuple(item.embedding) for item in negative_response.data
                ]
                self._negative_trait_embeddings = list(
                    zip(self._trait_descriptions.keys(), negative_embeddings)
                )

    async def score_open_answers(
        self,
        answers: Sequence[str],
        *,
        question_prompts: Sequence[str] | None = None,
    ) -> Mapping[str, float]:
        if not answers:
            return {}
        await self._ensure_embeddings()
        if not self._trait_embeddings:
            return {}
        embedding_inputs = _build_embedding_inputs(answers, question_prompts)
        response = await self._client.embeddings.create(
            model=self._model,
            input=embedding_inputs,
        )
        contributions: dict[str, float] = defaultdict(float)
        for answer_embedding in response.data:
            for idx, (trait, trait_embedding) in enumerate(self._trait_embeddings):
                score = _cosine_similarity(
                    answer_embedding.embedding, trait_embedding)
                if score >= self._threshold:
                    # Skalowalny boost - im bardziej ponad threshold, tym większy wkład
                    excess = score - self._threshold
                    boost_factor = 1.0 + \
                        (excess / (1.0 - self._threshold)) * 3.0  # Do 4x więcej
                    contributions[trait] += score * \
                        self._max_boost * boost_factor
                if self._allow_negative and self._negative_trait_embeddings:
                    negative_embedding = self._negative_trait_embeddings[idx][1]
                    negative_score = _cosine_similarity(
                        answer_embedding.embedding, negative_embedding
                    )
                    if negative_score >= self._threshold:
                        negative_excess = negative_score - self._threshold
                        negative_boost = 1.0 + (
                            (negative_excess / (1.0 - self._threshold)) * 3.0
                        )
                        contributions[trait] -= (
                            negative_score
                            * self._max_boost
                            * negative_boost
                            * self._negative_weight
                        )
        return contributions

    async def score_open_answers_detailed(
        self,
        answers: Sequence[str],
        *,
        question_prompts: Sequence[str] | None = None,
    ) -> list[dict]:
        """Return detailed scoring information for each open answer."""
        if not answers:
            return []

        await self._ensure_embeddings()
        if not self._trait_embeddings:
            return []

        embedding_inputs = _build_embedding_inputs(answers, question_prompts)
        response = await self._client.embeddings.create(
            model=self._model,
            input=embedding_inputs,
        )

        details = []
        for answer_idx, answer_embedding in enumerate(response.data):
            answer_text = answers[answer_idx]
            answer_detail = {
                "answer_number": answer_idx + 1,
                "answer_text": answer_text[:100] + "..." if len(answer_text) > 100 else answer_text,
                "trait_matches": []
            }

            for idx, (trait, trait_embedding) in enumerate(self._trait_embeddings):
                score = _cosine_similarity(
                    answer_embedding.embedding, trait_embedding)
                if score >= self._threshold:
                    excess = score - self._threshold
                    boost_factor = 1.0 + \
                        (excess / (1.0 - self._threshold)) * 3.0
                    contribution = score * self._max_boost * boost_factor
                else:
                    contribution = 0

                answer_detail["trait_matches"].append({
                    "trait": trait,
                    "cosine_similarity": score,
                    "contribution": contribution,
                    "above_threshold": score >= self._threshold,
                    "match_kind": "positive",
                })

                if self._allow_negative and self._negative_trait_embeddings:
                    neg_embedding = self._negative_trait_embeddings[idx][1]
                    neg_score = _cosine_similarity(
                        answer_embedding.embedding, neg_embedding
                    )
                    if neg_score >= self._threshold:
                        neg_excess = neg_score - self._threshold
                        neg_boost = 1.0 + (
                            (neg_excess / (1.0 - self._threshold)) * 3.0
                        )
                        neg_contribution = -(
                            neg_score
                            * self._max_boost
                            * neg_boost
                            * self._negative_weight
                        )
                    else:
                        neg_contribution = 0

                    answer_detail["trait_matches"].append(
                        {
                            "trait": trait,
                            "cosine_similarity": neg_score,
                            "contribution": neg_contribution,
                            "above_threshold": neg_score >= self._threshold,
                            "match_kind": "negative",
                        }
                    )

            details.append(answer_detail)

        return details

    @staticmethod
    def _build_negative_description(trait: str, description: str) -> str:
        return (
            f"Przeciwieństwo cechy '{trait}'. Osoba nie wykazuje zachowań opisanych jako: "
            f"{description}. W praktyce oznacza to świadome unikanie lub brak kompetencji "
            f"w tych obszarach."
        )


class PersonaNarrativeGenerator:
    def __init__(
        self,
        persona_repository: UserPersonaProvider,
        *,
        model: str | None = None,
    ) -> None:
        settings = get_openai_settings()
        self._client = create_async_openai_client(settings)
        self._model = model or os.getenv(
            "OPENAI_RESPONSE_MODEL") or "gpt-5-mini"
        self._persona_repository = persona_repository

    async def generate_and_store(
        self,
        user_id: str,
        psychology_traits: Mapping[str, float],
        vocation_traits: Mapping[str, float],
        highlights: Sequence[str],
    ) -> str:
        system_prompt = (
            "You are a supportive career coach. Summarize the user's personality and "
            "vocational traits in exactly ten sentences. Focus on strengths, growth opportunities, "
            "and how those traits align with potential life choices."
        )
        summary_lines = [
            "Psychological traits:",
            _top_trait_summary(psychology_traits),
            "Vocational preferences:",
            _top_trait_summary(vocation_traits),
            "User highlights:",
            "; ".join(
                highlights) if highlights else "brak dodatkowych wypowiedzi",
        ]
        user_prompt = "\n".join(summary_lines)
        try:
            response = await self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning={"effort": "low"},
            )
            persona_text = getattr(response, "output_text", None) or ""
            persona_text = persona_text.strip()
            persona_text = _enforce_sentence_count(persona_text, 10)
        except Exception:
            persona_text = _fallback_persona_text(
                psychology_traits, vocation_traits)

        await self._persona_repository.save_persona(
            user_id, persona_text, persona_type="tests"
        )
        return persona_text

    async def generate_psychology_persona_only(
        self,
        user_id: str,
        psychology_traits: Mapping[str, float],
        highlights: Sequence[str],
    ) -> str:
        system_prompt = (
            "Jesteś specjalistą psychologicznym. Stwórz szczegółowy profil psychologiczny użytkownika "
            "w dokładnie dziesięciu zdaniach. Wykorzystaj wszystkie dostępne dane: wyniki testów (bez liczb konkretnych i bez wspominania o teście samym w sobie), odpowiedzi na pytania otwarte "
            "(co brakuje do szczęścia, wymarzony partner, wymarzony dzień, największe lęki) oraz kontekst tych pytań. "
            "Opisz konkretne zachowania, preferencje i wzorce myślenia. Podaj praktyczne wskazówki rozwoju. "
            "Nie używaj zwrotów typu 'użytkownik' - pisz obiektywnie o osobowości."
        )
        summary_lines = [
            "Cechy psychologiczne:",
            _top_trait_summary(psychology_traits),
            "",
            "Odpowiedzi na pytania otwarte:",
            f"• Co brakuje do pełni szczęścia: {highlights[0] if len(highlights) > 0 else 'brak odpowiedzi'}",
            f"• Wymarzony partner życiowy: {highlights[1] if len(highlights) > 1 else 'brak odpowiedzi'}",
            f"• Wymarzony dzień: {highlights[2] if len(highlights) > 2 else 'brak odpowiedzi'}",
            f"• Największe lęki w życiu: {highlights[3] if len(highlights) > 3 else 'brak odpowiedzi'}",
        ]
        user_prompt = "\n".join(summary_lines)
        try:
            response = await self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning={"effort": "low"},
            )
            persona_text = getattr(response, "output_text", None) or ""
            persona_text = persona_text.strip()
            persona_text = _enforce_sentence_count(persona_text, 10)
        except Exception as e:
            # logger.warning(f"Failed to generate psychology persona: {e}")
            # Fallback persona
            persona_text = (
                "Osobowość charakteryzuje się zrównoważonym podejściem do relacji społecznych, "
                "z tendencją do adaptacji między aktywnościami grupowymi a potrzebą prywatności. "
                "Wysoka motywacja do rozwoju osobistego przejawia się w systematycznym dążeniu do samodoskonalenia. "
                "Silne umiejętności emocjonalne umożliwiają nawiązywanie głębokich i autentycznych relacji interpersonalnych. "
                "Kreatywność wyraża się w poszukiwaniu innowacyjnych rozwiązań i nietypowych perspektyw na problemy. "
                "Sumienność pozwala na konsekwentne realizowanie długoterminowych celów i zobowiązań. "
                "Tendencja do samokrytycyzmu może być przekształcona w konstruktywną samoocenę poprzez praktyki uważności. "
                "Potencjał przywódczy ujawnia się w sytuacjach wymagających koordynacji grupowej i inspirowania innych. "
                "Stabilność emocjonalna stanowi solidną podstawę do radzenia sobie ze stresem i wyzwaniami życiowymi. "
                "Praktyczne wskazówki rozwoju obejmują regularne ćwiczenia mindfulness i budowanie świadomości emocjonalnej."
            )

        await self._persona_repository.save_persona(
            user_id, persona_text, persona_type="psychology"
        )
        return persona_text


class TestProcessingService:
    def __init__(
        self,
        repository: TestRepository,
        persona_repository: UserPersonaProvider,
        *,
        psych_open_classifier: OpenAnswerTraitClassifier,
        vocation_open_classifier: OpenAnswerTraitClassifier,
        persona_generator: PersonaNarrativeGenerator,
    ) -> None:
        self._repository = repository
        self._psych_open_classifier = psych_open_classifier
        self._vocation_open_classifier = vocation_open_classifier
        self._persona_generator = persona_generator
        self._persona_repository = persona_repository

    async def submit_psychology_test(
        self, payload: PsychologyTestRequest
    ) -> TestSubmissionResponse:
        _validate_lengths(payload.closed_answers,
                          PSYCHO_QUESTION_IMPACTS, "psychology")
        _validate_open_count(payload.open_answers,
                             expected=4, test_name="psychology")

        # Zbierz szczegółowe informacje o scoring
        scoring_logs = []
        closed_answers_scoring = {}
        question_details = []

        # Szczegóły scoringu zamkniętych odpowiedzi
        for i, (answer, impacts) in enumerate(zip(payload.closed_answers, PSYCHO_QUESTION_IMPACTS)):
            question_num = i + 1
            question_detail = {
                "question_number": question_num,
                "answer": answer,
                "impacts": []
            }

            for impact in impacts:
                normalized = _normalize_likert(answer, reverse=impact.reverse)
                contribution = impact.weight * normalized
                question_detail["impacts"].append({
                    "trait": impact.trait,
                    "weight": impact.weight,
                    "normalized_answer": normalized,
                    "contribution": contribution,
                    "reverse": impact.reverse
                })

                scoring_logs.append(
                    f"Pytanie {question_num}: odpowiedź {answer} → cecha '{impact.trait}' "
                    f"(waga: {impact.weight}, znormalizowane: {normalized:.3f}, "
                    f"wniosek: {contribution:.3f})"
                )

            question_details.append(question_detail)

        trait_scores = _score_closed_answers(
            payload.closed_answers,
            PSYCHO_QUESTION_IMPACTS,
            PSYCHO_TRAITS,
        )

        # Szczegóły scoringu otwartych odpowiedzi
        open_scores = await self._psych_open_classifier.score_open_answers(
            payload.open_answers,
            question_prompts=PSYCHO_OPEN_QUESTION_PROMPTS,
        )
        open_answers_details = await self._psych_open_classifier.score_open_answers_detailed(
            payload.open_answers,
            question_prompts=PSYCHO_OPEN_QUESTION_PROMPTS,
        )

        scoring_logs.append(
            f"Scoring zamkniętych odpowiedzi: {dict(trait_scores)}")
        scoring_logs.append(
            f"Scoring otwartych odpowiedzi: {dict(open_scores)}")

        merged = _merge_scores(trait_scores, open_scores)
        scoring_logs.append(f"Połączone wyniki: {dict(merged)}")

        await self._repository.save_psychology_test(
            payload.user_id,
            payload.closed_answers,
            payload.open_answers,
            merged,
        )

        # Generuj opis psychologiczny
        persona_text = await self._persona_generator.generate_psychology_persona_only(
            payload.user_id,
            merged,
            payload.open_answers,
        )
        scoring_logs.append("Wygenerowano opis psychologiczny")

        return TestSubmissionResponse(
            message="Zapisano wyniki testu psychologicznego.",
            trait_scores=dict(merged),
            persona_generated=persona_text is not None,
            persona_text=persona_text,
            closed_answers_scoring=closed_answers_scoring,
            open_answers_scoring=dict(open_scores),
            open_answers_details=open_answers_details,
            scoring_logs=scoring_logs,
            question_details=question_details,
        )

    async def submit_vocational_test(
        self, payload: VocationalTestRequest
    ) -> TestSubmissionResponse:
        if not await self._repository.has_psychology_results(payload.user_id):
            raise ValueError("Najpierw należy wypełnić test psychologiczny.")
        _validate_lengths(payload.closed_answers,
                          VOCATION_QUESTION_IMPACTS, "vocation")
        _validate_open_count(payload.open_answers,
                             expected=4, test_name="vocation")

        scoring_logs: list[str] = []
        closed_answers_scoring: dict[str, float] = {}
        question_details: list[dict] = []

        for i, (answer, impacts) in enumerate(
            zip(payload.closed_answers, VOCATION_QUESTION_IMPACTS)
        ):
            question_num = i + 1
            question_detail = {
                "question_number": question_num,
                "answer": answer,
                "impacts": [],
            }

            for impact in impacts:
                normalized = _normalize_likert(answer, reverse=impact.reverse)
                contribution = impact.weight * normalized
                question_detail["impacts"].append(
                    {
                        "trait": impact.trait,
                        "weight": impact.weight,
                        "normalized_answer": normalized,
                        "contribution": contribution,
                        "reverse": impact.reverse,
                    }
                )
                scoring_logs.append(
                    f"[Vocation] Pytanie {question_num}: odpowiedź {answer} → cecha '{impact.trait}' "
                    f"(waga: {impact.weight}, znormalizowane: {normalized:.3f}, wniosek: {contribution:.3f})"
                )

            question_details.append(question_detail)

        trait_scores = _score_closed_answers(
            payload.closed_answers,
            VOCATION_QUESTION_IMPACTS,
            VOCATION_TRAITS,
        )
        open_scores = await self._vocation_open_classifier.score_open_answers(
            payload.open_answers,
            question_prompts=VOCATION_OPEN_QUESTION_PROMPTS,
        )
        open_answers_details = await self._vocation_open_classifier.score_open_answers_detailed(
            payload.open_answers,
            question_prompts=VOCATION_OPEN_QUESTION_PROMPTS,
        )
        merged = _merge_scores(trait_scores, open_scores)
        scoring_logs.append(
            f"[Vocation] Scoring zamkniętych odpowiedzi: {dict(trait_scores)}"
        )
        scoring_logs.append(
            f"[Vocation] Scoring otwartych odpowiedzi: {dict(open_scores)}"
        )
        scoring_logs.append(f"[Vocation] Połączone wyniki: {dict(merged)}")
        await self._repository.save_vocational_test(
            payload.user_id,
            payload.closed_answers,
            payload.open_answers,
            merged,
        )
        psychology_traits = await self._repository.get_traits(
            payload.user_id, "psychology"
        )
        persona_text = await self._persona_generator.generate_and_store(
            payload.user_id,
            psychology_traits or {},
            merged,
            payload.open_answers,
        )
        return TestSubmissionResponse(
            message="Zapisano wyniki testu zawodowego.",
            trait_scores=dict(merged),
            persona_generated=bool(persona_text),
            persona_text=persona_text,
            closed_answers_scoring=closed_answers_scoring,
            open_answers_scoring=dict(open_scores),
            open_answers_details=open_answers_details,
            scoring_logs=scoring_logs,
            question_details=question_details,
        )


# --- Helpers -----------------------------------------------------------------


def _validate_lengths(answers: Sequence[int], impacts, test_name: str) -> None:
    if len(answers) != len(impacts):
        raise ValueError(
            f"Oczekiwano {len(impacts)} odpowiedzi zamkniętych dla testu {test_name}."
        )


def _validate_open_count(answers: Sequence[str], expected: int, test_name: str) -> None:
    if len(answers) != expected:
        raise ValueError(
            f"Oczekiwano {expected} odpowiedzi otwartych dla testu {test_name}."
        )


def _score_closed_answers(
    answers: Sequence[int],
    impacts: Sequence[Sequence[TraitImpact]],
    traits: Sequence[str],
) -> Mapping[str, float]:
    totals: dict[str, float] = defaultdict(float)
    caps: dict[str, float] = defaultdict(float)
    for answer, trait_impacts in zip(answers, impacts):
        for impact in trait_impacts:
            raw = _normalize_likert(answer, reverse=impact.reverse)
            totals[impact.trait] += impact.weight * raw
            caps[impact.trait] += impact.weight
    for trait in traits:
        caps.setdefault(trait, 1.0)
    normalized = _normalize_scores(totals, caps)
    return normalized


def _normalize_likert(value: int, *, reverse: bool) -> float:
    if reverse:
        value = 8 - value
    return (value - 1) / 6.0


def _merge_scores(
    base_scores: Mapping[str, float],
    bonus_scores: Mapping[str, float],
) -> Mapping[str, float]:
    merged: dict[str, float] = dict(base_scores)
    for trait, bonus in bonus_scores.items():
        merged[trait] = min(
            max(merged.get(trait, 0.0) + bonus, 0.0),
            1.0,
        )
    return merged


def _build_embedding_inputs(
    answers: Sequence[str],
    question_prompts: Sequence[str] | None = None,
) -> list[str]:
    inputs: list[str] = []
    for idx, answer in enumerate(answers):
        prompt = (
            question_prompts[idx].strip()
            if question_prompts and idx < len(question_prompts)
            else ""
        )
        cleaned_answer = answer.strip()
        if prompt:
            inputs.append(f"{prompt}\nOdpowiedź użytkownika: {cleaned_answer}")
        else:
            inputs.append(cleaned_answer)
    return inputs


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _top_trait_summary(traits: Mapping[str, float], top: int = 5) -> str:
    if not traits:
        return "brak danych"
    ordered = sorted(traits.items(), key=lambda item: item[1], reverse=True)
    summary = ", ".join(f"{name}:{value:.2f}" for name, value in ordered[:top])
    return summary


def _enforce_sentence_count(text: str, target: int) -> str:
    sentences = [s.strip() for s in re.split(
        r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) == target:
        return " ".join(sentences)
    sentences = sentences[:target]
    while len(sentences) < target:
        sentences.append(
            "Dodaję zdanie, aby zachować wymaganą długość wypowiedzi.")
    return " ".join(sentences)


def _fallback_persona_text(
    psychology_traits: Mapping[str, float],
    vocation_traits: Mapping[str, float],
) -> str:
    sentences = [
        "Twoje odpowiedzi pokazują, że dobrze znasz siebie i chcesz nad sobą pracować.",
        "Psychologicznie wyróżniają Cię cechy takie jak "
        + _top_trait_summary(psychology_traits, top=3),
        "Te jakości budują solidny fundament pod kolejne decyzje.",
        "Zawodowo najmocniej świecą obszary: " +
        _top_trait_summary(vocation_traits, top=3),
        "Łącząc obie perspektywy, widać spójność między tym, co lubisz a tym, co Ci naturalnie wychodzi.",
        "Doceniaj zarówno swoje talenty miękkie, jak i praktyczne kompetencje.",
        "Małe codzienne kroki w wybranym kierunku pozwolą Ci utrzymać motywację.",
        "Dawaj sobie prawo do odpoczynku i testowania nowych pomysłów bez presji perfekcji.",
        "Twoja determinacja i ciekawość świata sprawiają, że masz szerokie spektrum możliwości.",
        "Niech ten profil będzie inspiracją do dalszych działań – jestem tu, by Ci kibicować.",
    ]
    return " ".join(sentences)


def build_test_processing_service() -> TestProcessingService:
    client = create_supabase_async_client()
    repository = TestRepository(client)
    persona_provider = _build_persona_provider(client)
    psych_classifier = OpenAnswerTraitClassifier(
        trait_descriptions=PSYCHO_TRAIT_DESCRIPTIONS,
        threshold=0.46,
        max_boost=0.15,
        model=os.getenv("OPENAI_CATEGORY_MODEL"),
    )
    vocation_classifier = OpenAnswerTraitClassifier(
        trait_descriptions=VOCATION_TRAIT_DESCRIPTIONS,
        threshold=0.48,
        max_boost=0.12,
        model=os.getenv("OPENAI_CATEGORY_MODEL"),
    )
    persona_generator = PersonaNarrativeGenerator(
        persona_repository=persona_provider,
        model=os.getenv("OPENAI_RESPONSE_MODEL"),
    )
    return TestProcessingService(
        repository=repository,
        persona_repository=persona_provider,
        psych_open_classifier=psych_classifier,
        vocation_open_classifier=vocation_classifier,
        persona_generator=persona_generator,
    )


def _build_persona_provider(client) -> UserPersonaProvider:
    table_name = os.getenv("SUPABASE_USER_PERSONA_TABLE")
    try:
        return SupabaseUserPersonaRepository(client, table_name=table_name)
    except Exception:
        return NullUserPersonaProvider()
