# Context Aggregation And Quality Benchmark Design

**Goal:** Expand work context with AniList, Webnovel, and Fandom while using local PT-BR chapter references as a global quality benchmark for OCR, inpainting, typesetting, and translation.

**Architecture:** The app will split the problem into two independent layers. A context aggregation layer resolves the correct work, ingests free external sources, and condenses them into structured context for OCR review and translation. A quality benchmark layer uses local PT-BR examples as a product-wide reference for acceptable output quality without overfitting to a single series.

**Tech Stack:** React 19 + TypeScript frontend, Rust Tauri commands, Python pipeline, reqwest/urllib scraping, local reference assets in `exemplos/`.

---

## 1. Objectives

- Replace single-source AniList lookup with a multi-source work lookup.
- Support short-list selection when external sources return ambiguous titles.
- Ingest all free-access Webnovel chapters for context enrichment.
- Ingest relevant Fandom pages for aliases, factions, powers, and terminology.
- Keep the final context structured and bounded so translation remains deterministic.
- Use `exemplos/exemploptbr` as a global quality target, not as work-specific canon.

## 2. Source Strategy

### AniList
- Use as the editorial baseline.
- Extract: canonical title, synopsis, genres, initial characters, cover.

### Webnovel
- Use as the strongest narrative source when available.
- Extract: matching works, free chapter catalog, readable chapter text, recurring terms, relationships, and narrative tone.
- Ignore locked or inaccessible chapters.

### Fandom
- Use as a terminology and entity supplement.
- Extract: character names, aliases, factions, techniques, powers, settings, and glossary-like pages.
- Treat as lower-confidence than Webnovel for plot facts.

## 3. Context Model

The pipeline should normalize all external data into a structured context object:

- `sinopse`
- `genero`
- `personagens`
- `aliases`
- `termos`
- `relacoes`
- `faccoes`
- `resumo_por_arco`
- `memoria_lexical`
- `fontes_usadas`

Priority rules:
- narrative facts: Webnovel first
- editorial metadata: AniList first
- terminology/navigation help: Fandom first

## 4. User Flow

### Setup Search
- User types a work name in setup.
- Backend searches AniList, Webnovel, and Fandom.
- If there is one strong match, the app can preselect it.
- If there are multiple plausible matches, the UI shows a short list for manual selection.

### Context Build
- Once the work is chosen, the backend fetches and condenses source data.
- The frontend receives a merged context preview.
- The project context is updated without exposing raw source dumps.

## 5. Pipeline Integration

### OCR / Review
- Character names, aliases, and recurring terms become hints for contextual OCR correction.

### Translation
- Structured context feeds the translator as compact fields rather than a giant prompt blob.
- Glossary suggestions can be pre-filled from extracted terms.

### Benchmark Layer
- Local PT-BR examples define the output quality target for:
  - OCR precision
  - inpainting cleanliness
  - legibility and text fit
  - translation naturalness
- These examples must not become story canon for unrelated works.

## 6. Error Handling

- If Webnovel or Fandom fail, keep AniList as the fallback.
- If Webnovel/Fandom return multiple matches, require explicit user choice.
- If chapter ingestion is partial, keep partial context and record used sources.
- Never block translation because one source failed.

## 7. Testing Strategy

- Unit test normalization/parsing of aggregated source payloads.
- Unit test short-list search responses and match merging.
- UI test manual match selection in setup.
- Regression test that pipeline still runs when only AniList succeeds.
- Later visual benchmark pass against `exemplos/exemploen` and `exemplos/exemploptbr`.
