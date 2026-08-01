# Style Copy R5 — Materialization and Owner Geometry Remediation Design

**Date:** 2026-08-01

**Status:** Approved

**Scope:** automatic Python pipeline only; owner-scoped typesetting, copyback, QA, benchmark, and validation evidence

## Problem statement

The Style Copy V2 remediation made owner identity, raster hashes, atomic commit, and export-gate composition fail closed. The fresh holdout at `N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546` nevertheless remained Style, Functional, Inspection, and Overall `NO-GO`.

The failure is systemic:

- eight of nine exact owners retained English because their cleanup/render attempt rolled back;
- colored-card owners produced no render ink and failed with `render_contract_invalid:render was not completed`;
- other owners drew pixels but failed with `render_contract_invalid:style raster contract applied decision mismatch`;
- the only committed PT-BR owner used incomplete cross-tile geometry and visually changed typography class;
- the benchmark blocked with font top-1 `0.40`, top-3 `0.00`, and required category, owner, speech, Delta E, containment, and catastrophic-mismatch metrics absent.

R5 must correct pixel production without weakening any gate, accepting English silently, or adding page/work/chapter exceptions.

## Root cause model

### 1. Style decisions cross three materialization domains

`style_application_decision_v2.applied_attributes` currently mixes:

- layout-owned facts: font size, alignment, container, occupancy, safe geometry;
- font-owned facts: selected font identity, weight, width/slant metrics;
- raster-owned facts: fill, stroke, glow, shadow, gradient, rotation, and glyph transforms.

`_rasterize_v2_layers_from_render_plan()` asks the glyph compositor to report the entire mixed decision. Attributes that the compositor cannot observe are classified as unmaterialized before layout/font evidence can be merged. The renderer then returns an empty review raster, so the atomic commit sees `render_completed=False`.

### 2. Intent and observation are compared without attribute canonicalization

For owners that draw pixels, the raster contract compares raw decision values with runtime values. Font file names versus font identities, color encodings, enum aliases, and numeric representations can describe the same materialized result but compare unequal. Conversely, copying the requested value into the observed contract would falsely claim application. R5 therefore needs attribute-specific observation and canonical comparison, not relaxed equality and not request echoing.

### 3. Owner geometry is not always page-space complete

Cross-tile and connected owners can retain component/tile geometry as their render-safe region. A complete translated payload is then fit into an incomplete spatial owner, producing clipping or a false `ok` fit. The complete owner body must remain one semantic payload and render against one authoritative page-space geometry contract.

### 4. Benchmark enforcement expects metrics the producer does not emit

The scorer correctly fails closed when mandatory metrics are absent. The generator/report pipeline still needs deterministic production of per-attribute, per-category, per-owner, speech, color-distance, containment, and catastrophic-mismatch evidence with nonzero denominators.

## Considered approaches

### A. Relax atomic comparison

Accept runtime values when an attribute differs or is missing. This is small but unsafe: it can commit a fallback font, missing effect, or incomplete geometry while declaring the requested style applied. Rejected.

### B. Patch only known mismatched attributes

Add special cases for font names and selected colors. This could clear current failures but would repeat for every new attribute/backend and would not solve cross-tile geometry or missing benchmark metrics. Rejected.

### C. Domain-aware materialization with canonical observation

Build a typed observation from layout, selected font, and raster pixels; canonicalize each attribute; compare the observed materialization to the approved decision only at the owner commit boundary. Couple it to page-space owner geometry and measurable benchmark evidence. Selected.

## Architecture

### Materialization intent

The existing immutable visual profile remains the authority for approved and abstained attributes. R5 does not mutate the approved decision during rendering.

Each applied attribute is assigned exactly one materialization domain:

| Domain | Attributes |
|---|---|
| layout | `font_size_px`, `alignment`, `container`, safe geometry, occupancy |
| font | `font_name`, `font_weight`, `font_width`, `slant_tangent`, font-derived width/x-height |
| raster | `fill`, `stroke`, `multistroke`, `shadow`, `glow`, `gradient`, `rotation_deg`, `tracking_xh`, `width_scale`, `scale_y` |

The domain registry must reject unknown ownership and duplicate ownership.

### Materialization observation

After layout fit, font resolution, and glyph composition, the renderer creates an immutable owner observation containing:

- owner/page/profile/geometry identities and hashes;
- canonical observed values by domain;
- per-attribute status: `materialized | abstained | mismatch | unavailable`;
- evidence references: layout contract, resolved font identity, glyph/effect envelopes, pixel/color metrics;
- final render status and reason.

An attribute is `materialized` only when its owning domain measured the applied result. Requested values are never copied into observed fields merely to satisfy equality.

### Canonical comparison

Comparison is attribute-specific:

- font identity uses the resolved file hash plus normalized family/style aliases;
- colors use normalized sRGB and Delta E where the contract specifies perceptual tolerance;
- numeric geometry uses finite normalized values and explicit tolerances;
- structured effects compare normalized color, width, radius, and offset fields;
- enums use a versioned alias map;
- containers compare page-space geometry identity and safe-region hash.

The raster contract records both the approved canonical intent and canonical observation. A mismatch remains fail closed and names the attribute/domain/evidence instead of returning the generic `applied decision mismatch`.

### Page-space owner geometry

Every renderable owner receives one `OwnerRenderGeometry` derived before inpaint from:

1. the union of every owner component;
2. complete selected-observation polygons;
3. trusted balloon/container evidence;
4. all executor/context projections, converted to page space;
5. foreign-owner protection masks.

The contract contains the semantic body bbox, source-replacement region, safe polygon(s), connected subregions, component/projection IDs, and a deterministic hash. Tile projections may transport evidence, but may not reduce the page-space owner geometry. Connected and cross-tile text remains one semantic payload; layout may wrap it but may not split, duplicate, truncate, or independently translate fragments.

### Atomic commit

The commit sequence remains:

1. validate cleanup mutation and residual evidence;
2. validate completed render and render quality;
3. validate profile, geometry, font, raster, and pixel hash bindings;
4. compare approved intent with the complete materialization observation;
5. compose cleanup plus glyph pixels once;
6. commit or roll back the entire owner.

Rollback continues to preserve the original pixels, marks the owner `review_required`, emits a precise diagnostic, and blocks export. R5 changes successful materialization, not safety semantics.

## Functional-language contract

Translation remains one request/response per semantic owner. A translated payload cannot be considered delivered unless that same owner commits cleanup and PT-BR glyph pixels. Final-language QA pairs the source challenge with the final owner region and blocks:

- English residual;
- missing PT-BR render;
- partial source removal;
- duplicate source/translation;
- split or clipped semantic body.

An owner may abstain from style attributes while still rendering conservatively only when policy explicitly permits that fallback. It may not preserve English and count as Functional GO.

## Benchmark and QA

The synthetic benchmark and real owner matrix share canonical metric names and versioned formulas. The generator must emit:

- font top-1/top-3 with known denominators;
- per-category GO rate and sample counts;
- speech and total-owner GO rates;
- fill/stroke/effect Delta E and geometry errors;
- safe containment and decorative false-positive rate;
- catastrophic high-confidence mismatch count;
- abstention correctness and round-trip integrity.

Missing metrics, missing required categories, zero denominators, hash mismatch, or incomplete native inspection remain `BLOCK`.

## Validation strategy

Validation proceeds from narrow to broad:

1. pure unit tests for domain ownership and canonicalization;
2. renderer tests proving each domain reports only observed values;
3. atomic-commit tests for equivalent, mismatched, absent, and unavailable attributes;
4. cross-tile/connected-owner geometry and whole-body layout tests;
5. functional-language tests proving rollback cannot pass with English;
6. benchmark producer/scorer tests for every mandatory metric and denominator;
7. the exact nine-owner sentinel across three works;
8. style, functional-adjacent, invariance, and full differential suites;
9. a fresh external holdout with native six-panel inspection and separate Style, Functional, Inspection, and Overall verdicts.

## Acceptance criteria

R5 is visually acceptable only when all of the following are true:

- exact target resolution succeeds for all nine owners with unchanged source hashes;
- 9/9 owners commit translated glyph pixels and have nonempty render plans;
- zero `owner_execution_rollback` and zero silent font/style fallback;
- zero English residual, duplicate translation, incomplete inpaint, clipped body, or semantic split;
- profile, geometry, raster, and observation coverage are 100%;
- font top-1 is at least 0.90 and top-3 at least 0.98;
- every required category is at least 0.85, speech GO at least 0.95, and total owner GO at least 0.90;
- fill Delta E median is at most 8 and p95 at most 12;
- stroke Delta E is at most 10 and thickness error is within the existing design tolerance;
- glow/shadow, rotation, width/slant/tracking, x-height, containment, and false-effect thresholds from the V2 contract pass;
- all nine native contact sheets receive independent Functional and Style `GO`;
- no new broad-suite failure appears relative to the frozen baseline;
- no page-, work-, or chapter-specific rule is added.

## Dirty-worktree policy

The active branch and dirty checkout are authoritative. Existing changes in `pipeline/strip/process_bands.py`, `pipeline/ownership/reconcile.py`, `pipeline/ownership/translation.py`, their tests, and unrelated generated/debug files belong to the user or earlier approved work.

R5 execution must:

- never use reset, checkout, restore, stash, clean, or a new worktree;
- inspect relevant diffs before every edit;
- add new modules where that reduces overlap;
- use `git add -p` for shared dirty files;
- verify cached diffs before every commit;
- keep `.codex-tmp` and external holdout evidence untracked;
- report any unavoidable overlap rather than overwriting local work.
