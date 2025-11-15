# Advice System Overview

## High-Level Flow
1. **Request intake (`/advice`)**
   - Accepts `user_id` or `Authorization` token and the latest user message (`message` query param).
   - Logs request metadata, validates presence of an identifier, and delegates to `AdviceService`.

2. **AdviceService**
   - Composes an `AdviceSelectionPipeline` with repositories, classifiers, and response generator.
   - For Supabase mode, builds:
     - `SupabaseAdviceRepository`
     - `SupabaseAdviceCategoryRepository`
    - `OpenAIEmbeddingCategoryClassifier`
    - `OpenAIEmbeddingAdviceIntentDetector`
    - `LLMAdviceResponseGenerator` (OpenAI LLM)

3. **AdviceSelectionPipeline**
   - **Category inference**
     - `OpenAIEmbeddingCategoryClassifier` embeds the message using `text-embedding-3-large`.
     - Produces the top 6 category matches (`CategoryMatch` objects) containing the OpenAI similarity score and rank (position in descending order).
     - Matches names against categories stored in Supabase (`advice_categories.name`) using a flexible variant map (supports lowercase, ASCII, `-`, `_`).
  - **Intent detection**
    - `OpenAIEmbeddingAdviceIntentDetector` embedduje wypowiedź do TOP5 wyników i porównuje z opisami każdego `AdviceKind`.
    - Gdy najwyższy wynik przekroczy próg (domyślnie 0.485) pipeline traktuje go jako prośbę o konkretny rodzaj, w przeciwnym razie przyjmuje pełną swobodę wyboru.
   - **Candidate retrieval**
     - If preferred kind is present, fetches advices of that kind filtered by matched categories, otherwise by overlap, falling back to the entire catalogue.
  - **Ranking & selection**
    - Category usage frequencies are lazily cached (one pass on all advices) so we can reward rare categories.
    - For each candidate the pipeline:
      - Computes a specificity factor `(max_item_categories + 1) / (category_count + 1)` (fewer categories ⇒ stronger weight).
      - Aggregates contributions from every overlapping `CategoryMatch` using `(similarity²) × ranking_weight × rarity_weight² × specificity_factor`.
        - `ranking_weight` leverages the classifier order (1…N).
        - `rarity_weight` uses `(total_advices + 1)/(frequency + 1)` and is additionally boosted when a category is unique and sits in TOP2.
      - Applies intent multipliers (`×1.8` when the advice format matches the user’s request, `×0.15` otherwise).
      - Adds jitter (±15%) to avoid deterministic behaviour.
    - If a candidate is the sole advice containing the top-ranked category (frequency = 1 and rank = 1) it is returned immediately (100% probability). Rank 2 still receives a very high boost.
    - Otherwise applies a weighted random choice across all candidates based on the computed weights, ensuring a mix of determinism and variety.
  - **Response rendering**
    - `LLMAdviceResponseGenerator` pobiera opis osobowości użytkownika (jeśli istnieje), łączy go z detalami porady i generuje dokładnie 10 zdań w tonie troskliwym.
    - `AdviceRecommendation` zawiera zarówno domenowy obiekt porady, jak i wygenerowaną wiadomość.

4. **Response serialization**
   - `AdviceResponsePayload` (Pydantic) exposes the advice details and chat response.

## Data Layer
### Advice Repository (`SupabaseAdviceRepository`)
- Retrieves full advice records with joined categories (`advice_category_links → advice_categories.name`).
- Supports:
  - `get_all()`
  - `get_by_kind(kind)`
  - `get_by_kind_and_containing_any_category(kind, categories)` – filters by advice kind and any of the provided category names.
- Returns domain `Advice` objects with category names exactly as stored in Supabase.

### Category Repository (`SupabaseAdviceCategoryRepository`)
- Returns the list of category names (`advice_categories.name`) ordered alphabetically.
- Simple membership checks reuse the cached list instead of additional queries.

### User Persona Provider
- `SupabaseUserPersonaRepository` odczytuje tekstowy opis osobowości (`persona_text`) dla `user_id` (domyślna tabela `user_personas`), zwracając `None`, jeśli wpis nie istnieje.
- `NullUserPersonaProvider` to bezpieczny fallback, jeżeli dane nie są dostępne.

## Classifiers & Scoring Rules
### Category Matches
- `CategoryMatch` has `name`, `score` (embedding cosine similarity), and `rank` (1 = most relevant).
- Pipeline considers TOP 6 matches and logs all scores.

### Intent Matches
- `AdviceIntentMatch` niesie `kind` oraz `score`. Wpływa na wagi w selekcji (trafienie w proszony format ×1.8, mis-match ×0.15).
- Brak prośby ⇒ pipeline działa wyłącznie na kategoriach.

### Rarity & Ranking Influence
- Category rarity: advices that are the unique holder of a HIGH-ranked category are prioritised (rank=1 ⇒ deterministic win, rank=2 ⇒ very high weight).
- Rank weight: `(len(matches) - rank + 1) / len(matches)`; higher-ranked categories contribute more.
- Specificity factor: `(max_item_categories + 1) / (len(advice.categories) + 1)` rewards narrower advices without penalising broad ones excessively.
- Random jitter ensures repeated identical requests can yield slightly different valid answers.

## Configuration & Limits
- **Max classifier categories**: 6 (TOP6 considered in scoring).
- **Max categories per advice**: pipeline assumes rare cases up to 7 (configurable via `max_item_categories`).
- **OpenAI settings**:
  - `OPENAI_API_KEY` (plus opcjonalnie `OPENAI_ORGANIZATION`, `OPENAI_PROJECT`).
  - `OPENAI_EMBEDDINGS_MODEL` – model domyślny.
  - `OPENAI_CATEGORY_MODEL` – (opcjonalnie) osobny model embeddings dla klasyfikacji kategorii.
  - `OPENAI_INTENT_MODEL` – (opcjonalnie) osobny model embeddings dla rozpoznawania intencji.
- **OpenAI settings**:
  - `OPENAI_API_KEY` (plus opcjonalnie `OPENAI_ORGANIZATION`, `OPENAI_PROJECT`).
  - `OPENAI_EMBEDDINGS_MODEL` – model bazowy na potrzeby embeddingów.
  - `OPENAI_CATEGORY_MODEL` – (opcjonalnie) model embeddingów dla kategorii.
  - `OPENAI_INTENT_MODEL` – (opcjonalnie) model embeddingów dla intencji.
  - `OPENAI_RESPONSE_MODEL` – model LLM do generowania odpowiedzi (domyślnie `gpt-5-mini`).
  - `OPENAI_REASONING_EFFORT` – poziom `reasoning.effort` przekazywany w wywołaniach OpenAI Responses (np. `low`, `medium`); domyślnie `low`.
- **Supabase settings**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, opcjonalnie `SUPABASE_USER_PERSONA_TABLE`.

### Tryby selekcji porady

- `ADVICE_SELECTION_MODE`
  - `categories` (domyślny): obecny tryb oparty o kategorie (`AdviceSelectionPipeline` + `OpenAIEmbeddingCategoryClassifier`).
  - `embedding`: nowy tryb oparty o embedding profilu użytkownika i treści porad (`PersonaEmbeddingAdviceSelectionPipeline`).
- `OPENAI_ADVICE_EMBEDDING_MODEL`
  - Nazwa modelu OpenAI używanego do embeddingów porad i profilu użytkownika (np. `text-embedding-3-large`).
  - Jeśli nie ustawiony, system użyje modelu z ogólnych ustawień (`OPENAI_EMBEDDINGS_MODEL`).

## Extensibility Notes
- Intent detection opiera się na `_OPENAI_INTENT_DEFINITIONS`; wystarczy zmienić listę opisów lub próg w `build_openai_intent_detector`.
- Response generation opiera się na `LLMAdviceResponseGenerator`; można podmienić prompt, model lub całą implementację.
- Category frequencies cache is computed on first use; invalidate manually if the catalogue changes frequently (e.g., inject a repository signal).
- Weighted selection can be tuned by adjusting rarity multipliers, jitter amplitude, or specificity formula.

## Operational Checklist
- Ensure Supabase tables:
  - `advices` with category links and kolumną `embedding` typu `real[]`:
    - przykładowe migracje:
      - `ALTER TABLE advices ADD COLUMN embedding real[];`
  - `advice_categories` containing human-readable `name`
  - `advice_category_links` mapping advices ↔ categories
- (Opcjonalnie) `user_personas` z kolumnami `user_id`, `persona_text`, `updated_at`.
- Populate `OPENAI_CATEGORY_DEFINITIONS` in `advice_service.py` to stay in sync with Supabase categories.
- Use `uvicorn app.main:app --reload --env-file .env` for local runs; `.env` should contain Supabase and OpenAI credentials.

