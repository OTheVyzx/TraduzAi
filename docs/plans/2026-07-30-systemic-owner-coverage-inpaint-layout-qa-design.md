# Systemic Owner Coverage, Inpaint, Layout, and QA Design

**Status:** Approved on 2026-07-30

## Context

The page-global owner architecture fixed the original band-authority problem, but the v10 systemic validation exposed gaps in the implementation and in the validation itself. The published report marked every matrix entry as `PASS`, while native-scale inspection still found:

- a card translated as `VALOR TOTAL DA COMPRA:` with `200 MILLION` left in English;
- owners whose rendered ink was only 24% to 59% of the source ink height while `fit_status` remained `ok`;
- pages with dozens of detector blocks, zero OCR records, `observation_complete=true`, and a passing export gate;
- residual source text, incomplete cleanup, duplicated or misplaced text, unsafe inpaint, and disproportionate layout in the broader visual sample;
- PT-PT lexical forms such as `mil milhões` when the configured target is PT-BR.

The first proven divergence for `200 MILLION` is between page-owner OCR and owner reconciliation. Strong observations contain `TOTAL PURCHASE AMOUNT 200MILLION`, but the reconciler selects the geometrically tighter truncated observation `TOTAL PURCHASE AMOUNT:`. The action mask and inpaint then correctly execute an incomplete authorization. This is not an inpaint-engine failure in isolation.

The design therefore treats functional correctness as an end-to-end contract. No single successful stage, exit code, or report is sufficient evidence of a valid export.

## Goals

- Preserve every material source-text observation until it has an explicit disposition.
- Produce one semantic owner, one translation, one cleanup, and one render per body.
- Keep bodies atomic through OCR, translation, layout, and rendering.
- Guarantee complete owner-scoped cleanup without modifying protected art.
- Eliminate band seams by making page-space owner execution authoritative.
- Reject text that is outside its container, overfilled, underfilled, duplicated, or disproportionate.
- Detect source-language residuals against the full source evidence set, not only the selected payload.
- Make inconclusive or internally inconsistent QA block export.
- Validate target-locale PT-BR independently from visual style.
- Solve categories of failure across works and chapters; never add page-, work-, title-, or band-specific rules.

## Non-goals

- Font matching, stroke, glow, shadow, gradient, slant, and other style-copy fidelity. Those belong to the separate Style Copy V2 design.
- Automatic rewriting of ambiguous translations.
- Hiding functional failures with a visually similar render.
- Restoring band-local authority for OCR, inpaint, layout, rendering, or QA.

## Considered approaches

### Upstream-only correction

Fix observation selection and mask construction. This would repair the known card, but a later regression or OCR failure could still pass undetected. It is insufficient by itself.

### QA-only correction

Strengthen the final gate without changing reconciliation or cleanup. This prevents bad exports but leaves the pipeline unable to produce a correct result. It is useful as a defense, not as the primary solution.

### Defense in depth — selected

Correct evidence selection, owner construction, mask authorization, inpaint verification, proportional layout, final observation, and export gating. Each layer independently detects violations relevant to its authority. This is the only approach that both produces correct pages and prevents silent regressions.

## Authority model

```text
page pixels
    -> source components + all OCR observations
    -> source-evidence ledger
    -> one semantic owner per body
    -> owner translation and PT-BR validation
    -> owner action/protection masks in page coordinates
    -> atomic pure-inpaint mutation
    -> proportional owner layout and glyph patch
    -> one page-space composition
    -> fresh independent final-pixel observation
    -> fail-closed export gate
```

Bands may schedule or project work, but they may not choose text, split a body, create an independent mask, normalize a separate background, render a duplicate, or decide export eligibility.

## 1. Source-evidence ledger

The ledger is derived from the canonical `SourceTextComponent` and `TextObservation` records. It is not a second OCR truth source. It records, per observation:

- associated component IDs and owner ID;
- normalized token stream used only for comparison;
- component coverage and observation precision;
- provider, confidence, language score, and provenance;
- materiality and corroboration;
- disposition: `selected`, `dominated`, `rejected_with_policy`, or `review_required`;
- reason and the observation that dominates it, when applicable.

No observation is removed merely because it was not selected. Strong, material, unselected evidence remains available to cleanup coverage and final-language QA.

Token normalization inserts letter-number boundaries, so `200MILLION` and `200 MILLION` compare consistently. It must not mutate payload text or arbitrary identifiers.

### Safe semantic dominance

A complete reading may dominate a truncated reading only when all conditions hold:

1. it covers at least the same semantic region;
2. the shorter normalized token stream is a coherent subsequence of the longer stream;
3. it does not cross another container, semantic role, or owner candidate;
4. geometry agrees or independent attempts/providers corroborate it;
5. the longer reading is not merely lower-precision contamination from adjacent content.

Selection ranks semantic/component coverage before tight IoU. Precision, language score, confidence, and provider remain tie-breakers and veto signals. “Longest text wins” is forbidden. Incompatible non-dominated readings make the owner `review_required`.

Component geometry is expanded from all polygons of the selected complete evidence before mask planning. Expansion after a truncated selection is not considered proof of full coverage.

## 2. Atomic semantic ownership

The existing page-global owner remains the only semantic execution unit.

Required invariants:

- every material source component has exactly one owner or an explicit preserve/review disposition;
- every body has exactly one source payload and one target payload;
- no owner is created from a line fragment when the lines form one body;
- repeated tile or band observations merge as evidence, not as additional bodies;
- one owner may render only once in the final glyph ownership map;
- connected balloons preserve one atomic payload and compatible shared layout constraints;
- cards may have distinct roles such as title, body, value, and footer, but a role body is never split to satisfy layout.

Owner identity and semantic hashes remain independent of style data.

## 3. Page-space mask and pure inpaint transaction

Mask planning consumes the selected owner evidence plus strong material corroborating polygons. Each selected line/polygon must contribute positive pixels to the action mask. Missing coverage for any selected line revokes the whole owner to review; there is no silent bounding-box fallback.

The transaction enforces:

```text
all selected source ink covered
changed_pixels subset_of action_mask
action_mask disjoint_from protected_art_mask
verified residual_score within threshold
one cleanup and one render for the same owner/hash chain
```

Negative operational evidence includes:

- glyphs and components owned by other owners;
- explicitly preserved text;
- character, icon, border, and line-art protection;
- connected foreground crossing the OCR support boundary.

If separation between source glyph and protected art is unreliable, the owner becomes `review_required`. The system must not expand the mask optimistically.

Inpaint and background normalization run once in page coordinates. Band crops are derived from that one result. This removes independent per-band color estimation and prevents visible seams or color shifts at band boundaries.

Residual verification is mandatory. An absent residual score is not success. A failed cleanup blocks glyph rendering and prevents the owner mutation from committing.

## 4. Proportional layout contract

Functional layout is responsible for readable scale, containment, centering, hierarchy, and atomic body placement. It does not copy decorative source style.

Every rendered owner produces an immutable `OwnerRenderQuality` contract containing at least:

```text
font_size_final
minimum_legible_font_px
source_ink_height_px
render_ink_height_px
source_scale_ratio
safe_height_occupancy
safe_area_occupancy
wrapped_line_count
containment_status
status
reasons
```

When trustworthy source polygons exist, their robust line/ink height is the primary proportional reference. A rendered/source ratio below `0.75` cannot be `ok`. Occupancy provides a second signal with thresholds calibrated by semantic role and payload length:

- short speech must not collapse into a tiny center mark;
- medium and long bodies use progressive occupancy bounds;
- titles and card roles use source scale and group hierarchy rather than blindly filling the container;
- connected regions use a common compatible scale;
- all rendered ink remains inside the safe polygon.

When no trustworthy source-scale evidence exists, verified container bounds and conservative role-specific limits are used. Missing quality evidence for a verified rendered owner is a blocking error.

The layout search may grow or shrink text to find a proportional fit, but it may not summarize, truncate, duplicate, or semantically split the body. If no candidate satisfies containment and proportional legibility, the owner returns `below_proportional_legibility` and the atomic transaction rolls back to review.

## 5. Independent final-pixel observation

The final observer must not reuse the semantic filtering that produced the page. It runs a raw final-pixel probe over the persisted artifact and records:

- detector blocks;
- OCR records;
- an OCR result or explicit failure reason for every material block;
- whole-page and owner/component-anchored crops;
- native-scale and selectively upscaled attempts;
- deduplicated tokens, geometry, provider, and confidence;
- artifact path, page ID, persisted SHA-256, and completion state.

`detected_blocks > 0` with zero usable OCR records is incomplete unless every block has a justified non-text disposition. Incomplete observation is always `BLOCK`, never `PASS`.

The observer must consume a real final-probe mode in the OCR runtime. Scanlation, SFX, cover, or other pipeline policies may be evaluated after raw evidence is collected; they may not suppress the evidence itself.

## 6. Final contracts and export gate

The gate evaluates the persisted final pixels after the last writer.

### Source coverage contract

Every material observation is semantically represented by the owner payload or has an explicit permitted disposition. Strong complete evidence cannot be hidden behind a truncated selected payload.

### Owner graph contract

Every component has one valid owner/disposition and every owner has one atomic execution path.

### Pixel ownership contract

Source-evidence ink is covered by that owner’s cleanup map, rendered glyphs belong to the expected owner, and no duplicate write occurs.

### Protected-art contract

No changed pixel intersects protected art and no mutation occurs outside the action mask.

### Residual cleanup contract

Every mutated owner has a verified acceptable residual score. Missing evidence blocks.

### Layout legibility contract

Every rendered owner has a complete `OwnerRenderQuality` record and satisfies containment, scale, and occupancy rules.

### Final language contract

Fresh final OCR is compared against:

- the target payload;
- the selected source payload;
- every strong material source observation associated with the owner/component.

This catches visible text that was omitted before translation, such as `200 MILLION`. Preserved SFX, names, and credits require an explicit policy and remain auditable.

### QA integrity contract

The observer is fresh, complete, page-correct, artifact-hash-correct, and internally consistent. Any absent or contradictory evidence blocks export.

## 7. PT-BR locale conformance

Locale validation is a content contract, not a style rule. It runs after translation and before owner rendering.

- deterministic lexical rules flag known target-locale mismatches, such as PT-PT `mil milhões` for PT-BR output;
- numeric meaning is preserved and checked separately from wording;
- high-confidence deterministic mismatches block or request review;
- ambiguous stylistic wording produces review evidence, not automatic rewriting;
- locale rules are configuration-driven and covered by neutral fixtures.

## 8. Debug and audit artifacts

Each page-owner run publishes enough evidence to reconstruct every decision:

- source-evidence ledger;
- selected/dominated observation trace;
- component geometry before and after safe expansion;
- action, protection, changed-pixel, and residual maps;
- layout candidates and final quality contract;
- owner glyph map;
- raw final detector/OCR attempts and completion reasons;
- contract-to-artifact links in the export gate.

The visual matrix report lists the exact pages, native-scale segments, and owner crops actually inspected. It may not infer visual approval from a successful runner or empty issue list.

## Error handling

The functional path is fail-closed at owner granularity and at export:

- ambiguous OCR: owner `review_required`;
- incomplete selected-source coverage: no mask execution;
- unsafe mask/protection overlap: no inpaint;
- unverified/high residual: no render or commit;
- no proportional layout: rollback owner mutation;
- incomplete final observation: export `BLOCK`;
- source text still visible: export `BLOCK`;
- PT-BR deterministic mismatch: block or review according to configured severity.

There is no legacy band fallback for a verified owner graph.

## Testing strategy

Implementation follows strict TDD with a failing contract test before each behavior change. Fixtures are category-based and work-neutral.

Required categories include:

- tight truncated reading versus corroborated complete numeric body;
- multiline card with header, value, body, and footer;
- one body observed across tiles/bands;
- duplicate observations and duplicate render attempts;
- source ink beside a character, icon, border, or connected line art;
- cross-band background susceptible to a color seam;
- incomplete mask and residual cleanup;
- short, medium, long, connected, card, and title layout;
- underfill, overflow, off-center, and out-of-safe-region render;
- detector blocks with empty OCR;
- strong unselected source evidence visible in final pixels;
- explicit SFX/name/credit preservation;
- PT-BR locale mismatch.

After focused tests, the suite runs owner property tests, pipeline integration tests, the systemic matrix, broad Python tests, and native-scale visual review. Calibration and holdout pages must be separated.

## Acceptance criteria

- Zero material source observations without an owner/disposition.
- Zero strictly truncated observations dominating corroborated complete evidence.
- Zero atomic bodies split through OCR, translation, layout, or render.
- Every selected line represented in the action mask.
- Zero changed pixels outside the action mask or inside protected art.
- Every cleanup has an acceptable verified residual score.
- Zero duplicate owner renders.
- Every rendered owner has a complete quality contract.
- No trustworthy owner with `source_scale_ratio < 0.75` marked `ok`.
- 100% rendered ink contained in its safe region.
- No final page with detector evidence and inconclusive OCR marked complete.
- No material source-language residual matched against the full source evidence set.
- No deterministic PT-BR locale violation in an exported page.
- All gates non-blocked across at least three works plus a holdout set.
- Native-scale human review records the exact inspected artifacts and returns GO.

## Relationship to Style Copy V2

This design owns content, geometry, cleanup, layout proportionality, and functional QA. Style Copy V2 may consume stable owner IDs, source masks, safe regions, and final layout bounds. It may not alter OCR, translation, owner identity, payload, mask, inpaint, or functional gate results.
