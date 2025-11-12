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
     - `EchoAdviceResponseGenerator` (placeholder)

3. **AdviceSelectionPipeline**
   - **Category inference**
     - `OpenAIEmbeddingCategoryClassifier` embeds the message using `text-embedding-3-large`.
     - Produces the top 6 category matches (`CategoryMatch` objects) containing the OpenAI similarity score and rank (position in descending order).
     - Matches names against categories stored in Supabase (`advice_categories.name`) using a flexible variant map (supports lowercase, ASCII, `-`, `_`).
  - **Intent detection**
    - `OpenAIEmbeddingAdviceIntentDetector` embedduje wypowiedź do TOP5 wyników i porównuje z opisami każdego `AdviceKind`.
    - Gdy najwyższy wynik przekracza próg (domyślnie 0.45) pipeline traktuje go jako prośbę o konkretny rodzaj, w przeciwnym razie przyjmuje pełną swobodę wyboru.
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
     - `AdviceRecommendation` wraps the advice domain object and a placeholder chat response (to be replaced by LLM-driven text).

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
- **Supabase settings**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.

## Extensibility Notes
- Intent detection opiera się na `_OPENAI_INTENT_DEFINITIONS`; wystarczy zmienić listę opisów lub próg w `build_openai_intent_detector`.
- Response generation currently echoes placeholders – replace with an LLM-backed generator for production.
- Category frequencies cache is computed on first use; invalidate manually if the catalogue changes frequently (e.g., inject a repository signal).
- Weighted selection can be tuned by adjusting rarity multipliers, jitter amplitude, or specificity formula.

## Operational Checklist
- Ensure Supabase tables:
  - `advices` with category links
  - `advice_categories` containing human-readable `name`
  - `advice_category_links` mapping advices ↔ categories
- Populate `OPENAI_CATEGORY_DEFINITIONS` in `advice_service.py` to stay in sync with Supabase categories.
- Use `uvicorn app.main:app --reload --env-file .env` for local runs; `.env` should contain Supabase and OpenAI credentials.

