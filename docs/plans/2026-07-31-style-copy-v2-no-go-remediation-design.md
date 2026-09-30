# Style Copy V2 NO-GO Remediation Design

**Status:** Approved direction on 2026-07-31

## Context

The first Style Copy V2 implementation completed its technical tasks, but the real held-out run is a visual and contractual **NO-GO**. This is not a tuning failure and must not be corrected with rules tied to a work, chapter, page, band, text literal, or coordinate.

The frozen validation root `.codex-tmp/style_copy_v2_validation_20260731_195900` contains 43 rendered owners across colored cards, cross-tile cases, and dark panels. All 43 owners have a `visual_profile_v2`, but all 43 profiles are `fallback`, no style attribute was applied, 559 attributes were abstained, and no owner persisted a `style_v2_raster_contract`. Nevertheless, the style and export gates reported `PASS`.

The source confidence is not generally absent: 42 of 43 rendered owners have at least one selected OCR observation with confidence at or above 0.70. The confidence and visual provenance are lost at an internal boundary before the style decision is made.

The remediation therefore has two responsibilities:

1. make trustworthy source style evidence reach and control the pixels that are actually committed; and
2. make QA prove that chain end-to-end, failing closed when evidence, execution, coverage, or provenance is missing.

## Goals

- Capture owner-scoped style evidence from canonical pre-inpaint pixels and authoritative masks.
- Preserve candidate eligibility without adding visual data to the translation payload or semantic owner identity.
- Decide every style attribute independently from its own evidence and confidence.
- Match local fonts from the real source text and source glyph silhouette.
- Materialize and rasterize all approved attributes without silently dropping unsupported fields.
- Carry an immutable raster contract with the exact glyph pixels through atomic owner commit and project persistence.
- Aggregate multi-region rendering without losing segment contracts.
- Make unsupported curved rendering explicit and fail closed.
- Audit profile, decision, raster contract, committed pixels, coverage, and category thresholds by exact owner.
- Keep functional correctness and style fidelity as separate subgates with an explicit aggregate verdict.
- Produce a reproducible, owner-targeted holdout run and mandatory native-scale inspection.

## Non-goals

This remediation does not repair OCR text, translation, residual English, duplicated or partial PT-BR, owner coverage, inpaint residue, layout overflow, text splitting, or body proportionality. Those remain functional-pipeline responsibilities and may independently block export.

Style data must not:

- alter source text, translated text, semantic hashes, owner identity, owner count, route, or component membership;
- enlarge an action mask, weaken a protection mask, or authorize inpaint;
- alter the functional safe region or make a functionally blocked page exportable;
- treat names or remaining English as evidence of style success or failure;
- introduce a per-page exception or visual special case.

## Root-cause model

### A. Evidence and eligibility discontinuity

`pipeline/ownership/translation.py` intentionally emits a clean translation payload. `execute_owner_page_graph()` later rebuilds records from that payload and supplies those reduced records to `build_owner_visual_profiles()`. Confidence, promotion provenance, source masks, and V2 field evidence are absent there. `_style_evidence()` consequently constructs an empty legacy adapter, and `decide_style_copy_v2()` selects global fallback before per-attribute gates.

The runtime owner path does not call `extract_text_style_evidence_v2()`. `FontShapeMatcher` is present but is not connected to owner profile construction. This explains why real OCR confidence exists while every field reports `candidate_confidence_missing_or_low`.

### B. Evidence and materialization schema discontinuity

The canonical attribute set does not include all geometry controls already supported by the renderer. Tracking and slant are measured only as provenance, and `_materialize_style()` maps only a subset of approved fields. Width, weight, tracking, slant, and scale may therefore be measured or approved but still disappear before rasterization.

### C. Raster persistence discontinuity

The renderer adds `style_v2_raster_contract` to a deep-copied render block. `OwnerGlyphPatch` carries pixels, masks, geometry, and quality, but not that contract. `process_bands.py` reconstructs the final record from the original layout input, so the renderer mutation is lost.

Multi-region aggregation does not preserve child contracts. The curved-text branch returns before the V2 raster path and can emit pixels without a V2 contract.

### D. QA and gate discontinuity

The real run had no raster contracts and no applied attributes, yet style QA passed. The audit compares legacy materialized keys to canonical raster keys, treats a missing raster object as an empty acceptable fallback, and skips layers with missing or mismatched profiles. There is no complete owner denominator, schema/hash validation, or mandatory category coverage.

The matrix defaults style to `shadow`, silently clamps unavailable page selectors to the last page, labels whole pages as categories, and can therefore claim SFX, burst, or dark-panel coverage when the selected page contains no matching owner. Benchmark thresholds exist but are not authoritative, missing category metrics do not block, inspection is optional, and the benchmark runner always exits successfully.

## Selected architecture

```text
canonical source page
  + selected owner observations
  + owner glyph/support/context masks
  + OCR/SFX eligibility provenance
                |
                v
       OwnerStyleCapture sidecar
                |
      +---------+----------+
      |                    |
      v                    v
StyleEvidenceV2      FontShapeMatcher
      |                    |
      +---------+----------+
                v
     per-attribute StyleDecisionV2
                |
                v
        complete materialization
                |
                v
       capability-aware rasterizer
                |
                v
 OwnerGlyphPatch + immutable raster contract
                |
                v
 atomic validation / commit / persistence
                |
                v
 owner-scoped style audit + pixel metrics
                |
                v
 style subgate ---- functional subgate
                \       /
                 aggregate export gate
```

The visual sidecar is created before translation and inpaint, keyed by `owner_id`, and remains outside semantic serialization. Translation continues receiving only translatable content.

## 1. Owner style capture and eligibility contract

Introduce a typed `OwnerStyleCapture` (or equivalently named immutable record) for each renderable owner. It contains:

```text
schema_version
page_id
owner_id
component_ids
route_action
source_artifact_sha256
source_text_sha256
glyph_mask_sha256
context_mask_sha256
effect_region_mask_sha256
candidate_kind
candidate_confidence
candidate_confidence_provenance
promotion_status
promotion_provenance
style_evidence_v2
font_match_evidence
capture_sha256
```

Candidate eligibility is derived from selected owner observations, not from the clean translation record:

- primary OCR uses a deterministic aggregation of selected observation confidences and records every contributing observation ID;
- promoted SFX requires explicit promotion state and promotion provenance from the owner graph;
- raw, rejected, review-only, or unpromoted SFX never becomes style-eligible;
- confidence absent or non-finite means ineligible, never assumed confidence;
- changing capture data cannot change semantic owner hashes or translation payloads.

The capture is made from the canonical page before inpaint. Source text is frozen solely for font matching and hashed; it does not become translated content.

## 2. Authoritative masks and source extraction

`extract_text_style_evidence_v2()` receives the original RGB page and explicit masks:

- glyph core: pixels attributable to the selected owner's source glyphs;
- stroke ring: a derived ring outside the core, clipped to owner support;
- context: owner-local background excluding glyph/stroke/effect masks and every foreign owner mask;
- effect envelope: bounded region in which glow or shadow may be measured;
- negative evidence: clean context used to reject false effects.

All masks are canonical `uint8`, page-shaped, contiguous, hash-bound, and non-overlapping according to tested invariants. Context expansion is relative to x-height and clipped to the owner/component geometry; it must not sample card art or neighboring text as a glyph effect.

Every measured field produces `FieldEvidence` with value, confidence, provenance, measurement units, support count, and an explicit abstention reason when unavailable. Tracking, slant, width scale, vertical scale, and weight become first-class attributes rather than debug-only provenance.

## 3. Font matching

`FontShapeMatcher` becomes part of the owner capture path. It receives the frozen source text, source glyph mask, normalized rotation/slant/scale evidence, semantic role, and a versioned local font catalog.

The match result records:

- catalog and matcher versions;
- eligible font IDs and license/provenance;
- top-k scores;
- absolute top-1 score;
- margin over the second candidate;
- normalization parameters;
- cache key;
- selected font or an explicit ambiguity/no-candidate abstention.

No font is downloaded at runtime. A role prior may break a close tie only within a documented bound; it cannot turn poor shape evidence into high confidence. Cache keys include catalog version, normalized source text, glyph-mask hash, and normalization evidence.

## 4. Per-attribute decision and complete materialization

The candidate gate determines whether the profile may be considered. Once eligible, each attribute uses only its own evidence threshold and provenance:

- `font_name`;
- `font_weight`;
- `font_width` / `width_scale`;
- `font_size` / x-height target;
- `fill` and optional gradient;
- `stroke_color` and normalized stroke width;
- `shadow_color`, radius, opacity, and offset;
- `glow_color`, radius, and opacity;
- `rotation_deg`;
- `slant_tangent`;
- `tracking_xh`;
- `scale_x` and `scale_y`;
- case/alignment only where they are visual and do not change text semantics.

There is no confidence borrowing. Every decision includes requested value, confidence, threshold, provenance IDs, applied/fallback/review status, and reason.

Materialization must be total: each approved canonical field is either mapped to a rasterizer input or rejected before rendering because no advertised backend can apply it. An applied decision must never disappear silently. The backend contract advertises exact capabilities and the chosen backend is recorded.

## 5. Immutable raster contract

The renderer emits an `OwnerStyleRasterContract` together with the actual pixel patch. The contract is canonicalized and hashed after pixels are produced:

```text
schema_version
page_id
owner_id
visual_profile_sha256
profile_component_geometry_sha256
execution_component_geometry_sha256
source_artifact_sha256
source_glyph_mask_sha256
status: applied | fallback | review_required
backend
backend_version
capabilities
requested_attributes
applied_attributes
abstained_attributes
glyph_core_envelope
effect_envelope
render_metrics
segments
rendered_before_sha256
rendered_patch_sha256
rendered_after_sha256
contract_sha256
```

The two geometry hashes are intentionally distinct. `profile_component_geometry_sha256` binds the contract to the source-profile geometry schema. `execution_component_geometry_sha256` binds it to `OwnerGlyphPatch`/`OwnerMutation` execution geometry. They must not be directly compared to one another because the payload schemas differ.

`visual_profile_sha256` must equal the attached profile hash. `rendered_patch_sha256` is calculated from the emitted RGBA/RGB patch and mask. `rendered_after_sha256` is calculated from the committed owner result, not merely copied from renderer metadata.

## 6. Atomic transport and commit

`OwnerGlyphPatch` carries the immutable raster contract as a required field for every rendering route subject to Style V2. Test fixtures use one shared valid contract builder to avoid weak placeholder dictionaries.

Before atomic commit, validation proves:

- patch owner/page identity matches the owner graph;
- profile and raster schema versions are supported;
- profile hash and source hashes match;
- execution geometry hash matches the patch/mutation geometry;
- requested decisions equal the contract's requested fields;
- all applied decisions appear in raster-applied attributes;
- every omitted field has an explicit abstention/capability reason;
- patch hash matches the actual patch pixels;
- final hash matches the candidate committed pixels;
- core and effect envelopes respect their allowed regions.

Any missing, malformed, stale, or tampered contract rejects the whole owner mutation. Inpaint and typeset roll back together, the owner transitions to `review_required`, and the exact rejection reason is persisted. A contract enters `final_records` only with a successful atomic commit.

## 7. Multi-region and curved rendering

For multi-region owners, each region creates a child segment contract with segment ID, component IDs, bbox, applied attributes, metrics, and pixel hash. The parent contract contains ordered child hashes and aggregate metrics. Aggregation is deterministic and fails if a child is missing, duplicated, overlaps illegally, or disagrees with owner/profile identity.

Curved text must not bypass Style V2. Until the curved backend supports the complete raster contract, a curve request is an explicit `review_required` or documented fallback with no V2-applied attributes. It cannot silently claim `applied` or commit contract-free pixels.

## 8. Canonical style QA

The style audit derives its denominator from renderable owners in the execution graph, not from whatever layers happen to contain style metadata. For each eligible owner it validates:

- owner profile presence and schema;
- capture, evidence, and profile hashes;
- raster contract presence, schema, and self-hash;
- owner/page/profile/execution binding;
- decision-to-raster equality in canonical attribute names;
- raster-to-final pixel hashes;
- core/effect containment;
- per-attribute source-to-final metrics;
- explicit fallback and abstention reasons.

Missing profile, missing raster contract, missing required metric, zero denominator, or incomplete coverage cannot result in `PASS` under `enforce`.

A low-confidence attribute can legitimately abstain and remains measurable. A high-confidence requested attribute that is omitted or silently downgraded is a blocking mismatch. A backend change is not itself a mismatch when the resulting contract proves every requested attribute was applied.

Required metrics include font top-1/top-3, Delta E 2000 for fill/stroke/glow/shadow, normalized stroke/radius/offset, rotation, width, slant, tracking, occupancy, source/final x-height, safe containment, false effects, coverage, fallback rates, and catastrophic mismatches.

## 9. Gate composition

Functional and style results remain separate:

```text
qa.functional_export_gate
qa.style_fidelity.gate
qa.export_gate.subgates.functional
qa.export_gate.subgates.style
qa.export_gate.aggregate
```

The aggregate gate is normalized once after all subgates. `allowed`, status, issues, severities, and counts are recomputed rather than patched in place.

- functional `BLOCK` + style `PASS` = aggregate `BLOCK`;
- functional `PASS` + style `BLOCK` = aggregate `BLOCK`;
- audit exception or missing style report in `enforce` = `qa_integrity_failure` and `BLOCK`;
- final-language/English-residual failures appear only in the functional subgate;
- style divergence appears only in the style subgate;
- both can fail independently on the same owner.

Strict execution returns a failing code for an aggregate block. Normal execution may persist a blocked preview, but it must never label it exportable.

## 10. Owner-targeted matrix and reproducibility

The style matrix migrates to exact schema-v2 targets:

```json
{
  "entry_id": "holdout_work_case",
  "page_id": "page_028",
  "owner_id": "owner_...",
  "component_ids": ["component_..."],
  "category": "sfx",
  "split": "holdout"
}
```

There is no page clamping. A missing page, owner, component, source crop, final crop, category sample, or style report is a contract error. Categories are attached to owner crops, never inferred from a whole page. Each required category has an explicit minimum owner count and a work-separated holdout sample.

Style-specific configs force `style_copy_mode=enforce`; shared functional configs are not rewritten. The runner persists a `run_manifest.json` with Git HEAD, scoped diff hash, input/config/tool hashes, effective config, runtime lock, seed, exact commands, timestamps, return codes, and hashes of stdout, stderr, project, and final artifacts. `--validate-only` reuses and verifies this evidence instead of deleting it.

Inspection schema v2 is mandatory and owner-scoped. Each row contains native source/final dimensions, owner/page/category, bboxes, SHA-256, functional verdict/note, style verdict/note, and a derived overall verdict. Contact sheets show:

```text
source | masks/evidence | requested style | raster patch | final | safe-region overlay
```

Absence of completed inspection is `PENDING`, never `GO`.

## 11. Rollout

1. **Shadow capture:** build real sidecars, masks, evidence, matcher results, and QA reports without altering final pixels.
2. **Render canary:** enable complete materialization and raster contracts on calibration fixtures; atomic validation is already enforced.
3. **Enforced holdout:** run work-separated owner targets with style gate and mandatory inspection.
4. **Runtime enablement:** enable only after calibration and holdout criteria pass without semantic/mask/functional regressions.

Rollback is configuration-based: return style application to `shadow` while continuing capture/audit. Do not revert shared owner or functional fixes. Invalid or unsupported contracts remain fail-closed in every phase where V2 claims applied pixels.

## Testing strategy

Implementation follows strict RED-GREEN-REFACTOR TDD. The test layers are:

- typed capture/raster contracts and canonical hashing;
- translation-boundary and semantic-hash invariance;
- authoritative mask construction and negative evidence;
- real owner runtime extraction and font matching;
- per-attribute decisions and total materialization;
- backend capability negotiation and raster goldens;
- single-region, multi-region, curved, and tamper/rollback paths;
- project persistence and round-trip;
- owner-denominator QA, pixel/hash validation, and gate composition;
- exact-target matrix, threshold enforcement, manifest reuse, and inspection schema;
- synthetic calibration, frozen style corpus, work-separated holdout, and native-scale visual review.

Every task runs focused tests first, then the adjacent regression set. Files with existing local changes are inspected before patching and staged by hunk only.

## Acceptance criteria

The implementation may be declared Style GO only when all of the following are true:

- 100% of renderable style-eligible owners have a valid profile and raster contract.
- 100% of committed raster contracts bind to the correct owner, profile, execution geometry, patch pixels, and final pixels.
- Zero silent `legacy`, empty-contract, or unsupported-capability fallback.
- Zero high-confidence requested attribute omitted without a blocking result.
- Zero semantic payload, translation, owner, route, mask, inpaint, or functional-gate change caused by the style sidecar.
- Font top-1 is at least 90%, top-3 at least 98%, and no required category is below 85%.
- Fill median Delta E 2000 is at most 8 and p95 at most 12.
- Stroke Delta E 2000 is at most 10; thickness error is at most 20% of x-height or 1 px.
- Glow/shadow Delta E 2000 is at most 12; radius/offset error is at most 25% or 2 px.
- Rotation error is at most 3 degrees.
- Width, slant, tracking, and occupancy error are each at most 15%.
- When the functional container permits, final/source x-height remains between 0.85 and 1.15.
- Core containment is 100%; effect pixels remain inside the declared effect envelope.
- False decorative effects in simple balloons are at most 2%.
- Card groups use a consistent family/effect class and stroke variation is at most 1 px.
- Owner-level visual GO is at least 90% overall, ordinary speech at least 95%, and no required category below 85%.
- Every required category has at least its configured holdout owner count; zero-owner categories block.
- Zero catastrophic high-confidence mismatch in holdout.
- Native-scale inspection explicitly approves every required sample.

These criteria yield three verdicts:

- **Style verdict:** evaluates only style-copy correctness on functionally valid frozen-text targets.
- **Functional verdict:** evaluates coverage, text, language, inpaint, and layout independently.
- **Overall verdict:** GO only when both verdicts and mandatory inspection are GO.

A style-only frozen corpus may become GO while the live functional matrix remains NO-GO. Reports must say so explicitly; neither result may conceal the other.

## Definition of done

The remediation is complete only when the implementation, focused and adjacent suites, broad differential suite, fresh holdout run, benchmark scoring, owner-level contact sheets, native inspection, run manifest, and versioned report all exist and agree. An exit code or unit-test pass alone is not completion. If any acceptance criterion fails, publish the measured NO-GO and its owner-scoped evidence; do not add page-specific exceptions.
