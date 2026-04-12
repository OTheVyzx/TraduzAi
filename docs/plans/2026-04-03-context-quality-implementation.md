# Context Aggregation And Quality Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add unified work search and context enrichment from AniList, Webnovel, and Fandom, with UI short-list selection and pipeline-ready structured context.

**Architecture:** Rust will own external work lookup and source aggregation for the interactive setup flow, while Python will consume the enriched context as structured input during pipeline execution. The first implementation wave will focus on robust lookup, selection, and context merging; the quality benchmark layer will remain documented and partially wired for later tuning.

**Tech Stack:** Tauri Rust commands, reqwest + scraper-style HTML parsing, React + Zustand, Python pipeline context handling.

---

### Task 1: Add shared frontend types for aggregated work search

**Files:**
- Modify: `T:\mangatl\src\lib\tauri.ts`
- Modify: `T:\mangatl\src\lib\stores\appStore.ts`

**Step 1: Write the failing test**

Document expected shapes in type-safe call sites by introducing:
- `WorkSearchCandidate`
- `WorkSearchResponse`
- extended `ProjectContext`

**Step 2: Run test to verify it fails**

Run: `npm run build`
Expected: FAIL because the new types are referenced but missing.

**Step 3: Write minimal implementation**

Add the missing types and update exported bindings.

**Step 4: Run test to verify it passes**

Run: `npm run build`
Expected: PASS for the new type surface.

### Task 2: Implement Rust search aggregation command

**Files:**
- Modify: `T:\mangatl\src-tauri\src\commands\pipeline.rs`
- Modify: `T:\mangatl\src-tauri\src\lib.rs`

**Step 1: Write the failing test**

Add Rust tests for:
- merging AniList/Webnovel/Fandom candidates
- sorting strong matches ahead of weak matches
- returning bounded short-list results

**Step 2: Run test to verify it fails**

Run: `cargo test search_work -- --nocapture`
Expected: FAIL because the command/helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- `search_work_sources(query)`
- Webnovel result scraping
- Fandom result scraping
- candidate normalization
- `search_work` Tauri command returning a short list

**Step 4: Run test to verify it passes**

Run: `cargo test search_work -- --nocapture`
Expected: PASS

### Task 3: Implement Rust context enrichment command

**Files:**
- Modify: `T:\mangatl\src-tauri\src\commands\pipeline.rs`

**Step 1: Write the failing test**

Add Rust tests for:
- merging AniList data with external source details
- skipping locked/unreadable Webnovel chapters
- deduplicating characters/aliases/terms

**Step 2: Run test to verify it fails**

Run: `cargo test enrich_work_context -- --nocapture`
Expected: FAIL because the enrichment helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- `enrich_work_context(selection)`
- Webnovel chapter/catalog fetch with free-only filtering
- Fandom page fetch and term extraction
- bounded structured context output

**Step 4: Run test to verify it passes**

Run: `cargo test enrich_work_context -- --nocapture`
Expected: PASS

### Task 4: Wire short-list selection into setup

**Files:**
- Modify: `T:\mangatl\src\pages\Setup.tsx`
- Modify: `T:\mangatl\src\lib\tauri.ts`

**Step 1: Write the failing test**

Define the interaction:
- search query returns multiple candidates
- UI shows candidate cards
- user picks one
- project context updates from enriched payload

**Step 2: Run test to verify it fails**

Run: `npm run build`
Expected: FAIL because the state and handlers are missing.

**Step 3: Write minimal implementation**

Add:
- new search handler
- short-list UI
- loading/error states
- apply-selection handler

**Step 4: Run test to verify it passes**

Run: `npm run build`
Expected: PASS

### Task 5: Let Python preserve enriched context

**Files:**
- Modify: `T:\mangatl\pipeline\main.py`
- Modify: `T:\mangatl\pipeline\translator\context.py`

**Step 1: Write the failing test**

Create a Python test ensuring:
- provided enriched context is preserved
- AniList fallback still works when no enriched context exists

**Step 2: Run test to verify it fails**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest discover -s tests`
Expected: FAIL because the richer context shape is not handled.

**Step 3: Write minimal implementation**

Accept extra fields in the context dict and avoid overwriting them during fallback.

**Step 4: Run test to verify it passes**

Run: `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest discover -s tests`
Expected: PASS

### Task 6: Verify end-to-end safety

**Files:**
- Modify: `T:\mangatl\context.md`

**Step 1: Write the verification checklist**

List:
- Rust tests
- Python tests
- `npm run build`
- one manual or automated setup search flow

**Step 2: Run verification**

Run:
- `cargo test`
- `T:\mangatl\pipeline\venv\Scripts\python.exe -m unittest discover -s tests`
- `npm run build`

Expected: PASS, with any known live-site fragility called out clearly.
