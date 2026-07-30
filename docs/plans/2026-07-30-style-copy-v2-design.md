# Style Copy V2 Design

**Status:** Approved on 2026-07-30

## Context

Style fidelity is a separate concern from functional correctness. The v10 page-owner validation showed that functionally rendered text frequently falls back to ComicNeue with `style_origin=legacy`, even when the source contains a condensed UI font, expressive burst lettering, multistroke, glow, shadow, rotation, or an SFX such as `DING`.

The owner path currently loses visual evidence between source extraction, owner translation, and rendering. Existing dictionaries also disagree on field names, and one high-confidence attribute can authorize unrelated low-confidence attributes. The current font detector compares a fixed sample against a very small preset set, while SFX rendering may override the selected typeface with OpenCV Hershey.

The Style Engine S4 work already provides valuable shadow-mode foundations on separate branches/worktrees:

- typed per-attribute `StyleEvidenceV2`;
- typographic foreground/core/stroke mask decomposition;
- mask-backed fill and stroke evidence;
- calibrated font ranking in shadow mode;
- synthetic benchmark baseline.

Those foundations must be reconciled with the current page-owner branch instead of reimplemented or blindly merged across the heavily diverged renderer.

## Goals

- Capture source visual evidence before inpaint and carry it to the final owner render.
- Keep style completely outside semantic owner identity and functional execution.
- Apply each attribute only when that attribute has sufficient evidence.
- Match local fonts using the actual source text and glyph silhouette.
- Reproduce fill, stroke, glow, shadow, gradient, rotation, slant, width, weight, tracking, and SFX scale where supported.
- Keep visual groups internally coherent without flattening title/body/value hierarchy.
- Make fallbacks, abstentions, and mismatches explicit and auditable.
- Validate style source-to-final by `owner_id` on calibration and holdout corpora.

## Non-goals

- Correct OCR, translation, English residuals, duplicated text, incomplete inpaint, body splitting, or unsafe layout.
- Alter owner identity, route, payload, action mask, protection mask, or functional gate.
- Download fonts at runtime.
- Apply uncertain effects to make a result appear more similar.
- Use opaque raster style transfer that destroys editability or deterministic rendering.

## Considered approaches

### Preset-only rendering

Map semantic categories to a few curated presets. This is stable and cheap, but cannot reproduce work-specific card, burst, or SFX typography. Presets remain useful only as explicit fallbacks.

### Neural or raster style transfer

Transfer the source raster appearance directly. It may look convincing, but it is difficult to constrain, edit, audit, or keep independent from content and geometry. It is outside this version.

### Evidence-first hybrid — selected

Extract typed per-attribute evidence, match a local font, apply trustworthy attributes, and fall back by semantic context for uncertain fields. This preserves determinism and editability while improving fidelity progressively.

## Boundary with the functional pipeline

```text
source page + owner components + source glyph masks
             |
             +-- pre-inpaint capture --> visual_profiles_by_owner

functional pipeline
    OCR -> owner -> translation -> mask -> inpaint -> proportional layout
                                                        |
visual profile sidecar ---------------------------------+
                                                        v
                                             Style Copy V2 renderer
                                                        |
                                             style fidelity audit
```

The functional design provides stable `owner_id`, correct PT-BR payload, source/component geometry, cleanup masks, safe region, and proportional layout. Style Copy V2 consumes those values read-only after layout resolution.

A style decision cannot:

- change text or line semantics;
- create, merge, or split owners;
- affect translation;
- enlarge the cleanup mask;
- override safe-region containment;
- turn a blocked functional page into GO.

## 1. Typed visual evidence contract

The S4 `StyleEvidenceV2` model becomes the canonical foundation. Each attribute is represented independently:

```text
FieldEvidence:
    value
    confidence
    provenance
    top_k
    margin
    abstention_reason
```

The normalized profile covers:

- font family and candidate ranking;
- weight and width/condensation;
- size/x-height reference;
- alignment and case;
- fill color and gradient;
- stroke color and relative thickness;
- shadow color, radius, and offset;
- glow color and radius;
- rotation and slant;
- tracking;
- horizontal and vertical scale;
- semantic visual role and container class.

The owner-facing sidecar stores:

```text
OwnerVisualProfile:
    owner_id
    source_artifact_sha256
    glyph_mask_sha256
    semantic_role
    group_id
    evidence_by_field
    decisions_by_field
    style_copy_status
```

`style_copy_status` is one of `applied`, `fallback`, `not_applicable`, or `review_required`. A renderable owner must always have an explicit status; `legacy` is not an acceptable silent state.

## 2. Confidence and eligibility policy

Eligibility and field application are distinct decisions.

- Main OCR text must pass the shared source-style confidence gate.
- SFX must first be promoted and then pass its confidence gate; a raw visual candidate is never enough.
- Review-only or low-confidence evidence may be stored for debugging but cannot be applied.
- Missing confidence means abstention.
- Every attribute is authorized only by its own confidence and provenance.
- High-confidence glow cannot authorize low-confidence color, font, stroke, or any other field.

The implementation reuses one shared policy from runtime through audit. Threshold values are configuration constants with tests; they are not duplicated across `main.py`, renderer, SFX, and reporting.

## 3. Owner-scoped visual sidecar

Visual capture occurs on canonical page pixels before inpaint. The sidecar is keyed by `owner_id` and is not added to the semantic owner hash.

Required invariants:

- every renderable owner has exactly one profile/status;
- permuting tiles, bands, or observation order does not change the profile;
- a single owner cannot receive conflicting profiles from different projections;
- changing the sidecar cannot change semantic owner serialization, translation, mask, route, or mutation authorization;
- profile artifacts are tied to source image and glyph-mask hashes.

Owner groups, such as one UI card, may share a group identity for consistency. Individual roles retain their hierarchy.

## 4. Masked source extraction

The extractor receives separate masks:

- core glyph/foreground mask;
- stroke ring;
- context/background mask;
- optional outer effect region.

Measurements are normalized to x-height or glyph size so they survive translation length and layout scale changes. The extractor estimates:

- robust fill color and optional gradient;
- stroke color and thickness;
- glow and shadow color, radius, opacity, and offset;
- rotation, slant, width, weight, and tracking;
- upper/lower case and alignment;
- source glyph occupancy and scale reference.

Negative evidence is as important as positive evidence. Simple white balloons must not acquire false glow, gradient, or stroke from the background or antialiasing fringe.

## 5. Font matching

The neural/current detector is only a shortlist provider. Final matching uses the actual frozen source text:

1. enumerate eligible local, licensed font candidates from the versioned catalog;
2. render the source text for each candidate;
3. normalize scale, rotation, slant, and width;
4. compare the rendered mask with the source glyph mask using silhouette IoU and contour distance;
5. combine the score with role/context priors;
6. require an absolute score and a margin over the second candidate;
7. abstain to a contextual fallback when ambiguous.

The result is cached by font-catalog version, text, and normalized profile hash. Fonts are never downloaded during runtime.

## 6. Context policy and group consistency

Context controls which fields are safe and which fallback applies:

- **ordinary balloon:** family, weight, width, case, tracking, and proportion; decorative effects require strong evidence;
- **card/UI:** condensed family, fill, stroke, glow, gradient, hierarchy, and group consistency;
- **burst/SFX:** expressive family, rotation, horizontal/vertical scale, stroke, glow, shadow, and occupancy;
- **dark panel:** contrast-aware fill and effects;
- **text over art:** conservative color/effect application to avoid background contamination.

Members of the same visual group share a compatible family and effect class. Role-specific size, weight, and emphasis remain allowed. A single balloon body cannot vary style per OCR line unless the source contains explicit, independently evidenced spans; span styling is out of scope for the initial rollout.

## 7. Unified glyph rasterizer

Text and SFX use one capability model. The renderer supports:

- real tracking;
- slant/skew;
- horizontal condensation/expansion;
- independent SFX scale X/Y;
- solid or gradient fill;
- proportional multi-stroke;
- shadow and glow layers;
- rotation around the resolved layout anchor.

The hard-coded Hershey override for Latin SFX is removed. SFX follows the same selected font and field decisions as other owner text.

Optional Python/Rust backends advertise capabilities. If an applied profile requires an unsupported effect, the pipeline selects a capable backend explicitly. It may not silently drop the effect.

## 8. Style fidelity QA

Style QA pairs all evidence by `owner_id`:

```text
source pixels -> glyph masks -> evidence -> decisions -> render plan -> final pixels
```

The audit resolves the real source image path regardless of PNG/JPG, records hash provenance, and reports each field as:

- `not_scanned`;
- `fallback`;
- `applied`;
- `mismatch`;
- `not_applicable`.

It produces a visual sheet containing source glyphs, masks/evidence, expected render, and final render. High-confidence applied mismatches may block the style gate. Intentional fallback or abstention is reported but is not mislabeled as extraction failure.

The functional gate always runs first. Style QA evaluates only functionally valid, eligible owners.

## 9. Corpus, benchmark, and rollout

The corpus is owner-paired and freezes correct text so OCR or translation defects do not contaminate style metrics. It covers:

- ordinary white balloon;
- translucent balloon;
- yellow and blue cards;
- black and red bursts;
- SFX such as `DING`;
- dark panel;
- text over art.

Each case records source artifact, `owner_id`, glyph mask, frozen source text, category, group, and field expectations. Calibration and holdout are separated by work/chapter.

Rollout phases:

1. **Shadow:** capture V2 evidence and compare with S4 baselines without affecting rendered pixels.
2. **Canary:** apply high-confidence fields to the calibration corpus and selected non-export runs.
3. **Enabled with fallback:** apply per-field policy to eligible owners; preserve explicit fallbacks.
4. **Style gate:** block only catastrophic or high-confidence mismatches after holdout targets remain stable.

## Existing S4 integration strategy

The existing S4 commits are treated as source material and benchmark history, not blindly cherry-picked as a stack onto the current renderer. Integration proceeds file by file and test first:

- preserve the synthetic benchmark baseline;
- port the typed `StyleEvidenceV2` contract;
- port typographic mask decomposition;
- port mask-backed fill/stroke evidence;
- port calibrated font ranking in shadow mode;
- adapt imports and APIs to the current owner pipeline;
- prove semantic and mask invariance before enabling runtime application.

No reset, checkout, stash, destructive cleanup, or new clean worktree is required. Existing local changes remain untouched, and scoped commits stage only Style Copy V2 files.

## Error handling

- missing profile/status: explicit fallback or block according to rollout phase;
- low confidence: abstain for that field;
- conflicting owner profiles: `review_required`;
- ambiguous font: contextual fallback;
- contaminated extraction: field abstention, not whole-profile invention;
- unsupported backend capability: choose a capable renderer or block the style application;
- source/final hash mismatch: invalidate audit evidence;
- functional page blocked: skip style eligibility and preserve the functional failure.

## Testing strategy

Implementation follows strict TDD:

- contract serialization and per-field confidence tests;
- owner sidecar semantic-invariance/property tests;
- mask decomposition and positive/negative extractor fixtures;
- font matcher ranking, margin, ambiguity, and cache tests;
- context/group consistency tests;
- rasterizer golden tests for tracking, slant, width, scale, stroke, glow, shadow, gradient, and rotation;
- Python/Rust capability negotiation tests;
- source-to-final owner audit tests;
- calibration and work-separated holdout matrix;
- native-scale human review of contact sheets.

Goldens contain correct frozen text. For example, a functional `DING`/`VING` OCR error belongs to the functional plan; the style golden evaluates the expected correct `DING` appearance.

## Acceptance criteria

- 100% of renderable owners have explicit `style_copy_status`.
- Zero silent `legacy` fallback.
- Zero semantic payload, owner, route, mask, or mutation change caused by style data.
- Font match top-1 at least 90% and top-3 at least 98%; no category below 85%.
- Fill median Delta E 2000 at most 8 and p95 at most 12.
- Stroke Delta E 2000 at most 10; thickness error at most 20% of x-height or 1 px.
- Glow/shadow Delta E 2000 at most 12; radius/offset error at most 25% or 2 px.
- Rotation error at most 3 degrees.
- Width, slant, and occupancy error at most 15%.
- When the functional container permits it, final/source x-height ratio remains between 0.85 and 1.15.
- 100% safe-region containment.
- False decorative effects in simple balloons at most 2%.
- Card groups use a consistent family/effect class; stroke variation at most 1 px.
- At least 90% owner-level visual GO overall, at least 95% for ordinary speech, and no category below 85%.
- Zero catastrophic high-confidence mismatch in holdout.
- Human native-scale review explicitly approves the inspected artifacts.

## Relationship to the functional design

The functional plan is implemented and validated first. Style Copy V2 begins from functionally correct owner output and may advance in shadow mode in parallel, but runtime application and style gating cannot precede stable functional content, cleanup, layout, and final-pixel contracts.
