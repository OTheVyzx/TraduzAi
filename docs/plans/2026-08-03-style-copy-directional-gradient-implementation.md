# Style Copy Directional Gradient Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect the source text's real linear-gradient colors and direction, then reproduce that gradient continuously over the complete translated glyph body without source- or band-specific rules.

**Architecture:** Introduce one canonical normalized linear-gradient model shared by extraction, Style V2 policy, rendering, and materialization audit. Fit a robust two-dimensional color field only from authoritative glyph-interior samples, preserve legacy two-color lists as vertical gradients, and evaluate the field over the union of translated glyphs rather than once per line.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Pillow, pytest, existing Style Evidence V2/materialization contracts, existing GPU pipeline.

---

## Execution constraints

- Work in the existing dirty checkout `N:\TraduzAI`.
- Preserve every unrelated local change. Do not use `reset`, `checkout`, `stash`,
  `clean`, or a replacement worktree.
- Before every edit batch, inspect `git status --short` and the scoped diff.
- Stage only paths named by the current task. Never stage `.codex-tmp`.
- Follow RED→GREEN→REFACTOR. A production edit is not allowed until the named
  failing test has been run and its failure is understood.
- Run the real pipeline only for chapter 39 band 006 after unit and integration
  tests pass.
- Use @systematic-debugging if observed evidence differs from the synthetic
  model, @test-driven-development for every code change,
  @traduzai-typesetting for renderer-specific checks, and
  @verification-before-completion before reporting success.

## Canonical visual contract

All new evidence uses this value shape:

```python
{
    "kind": "linear",
    "colors": ["#4A2497", "#08080A"],
    "stops": [0.0, 1.0],
    "start": [0.0, 0.0],
    "end": [1.0, 1.0],
    "coordinate_space": "glyph_bbox_normalized",
}
```

Legacy `['#RRGGBB', '#RRGGBB']` values canonicalize to a top-to-bottom field.
The start/end axis is evaluated once over the bounding box of the complete glyph
union. It is never restarted for individual lines.

### Task 1: Add the canonical directional-gradient model

**Files:**
- Create: `pipeline/typesetter/gradient_model.py`
- Create: `pipeline/tests/test_gradient_model.py`

**Step 1: Write failing canonicalization tests**

Add tests for:

```python
def test_legacy_two_color_gradient_becomes_vertical_canonical_field() -> None:
    assert canonicalize_linear_gradient(["#ff0000", "#0000ff"]) == {
        "kind": "linear",
        "colors": ["#FF0000", "#0000FF"],
        "stops": [0.0, 1.0],
        "start": [0.5, 0.0],
        "end": [0.5, 1.0],
        "coordinate_space": "glyph_bbox_normalized",
    }


def test_structured_gradient_is_rounded_and_preserves_direction() -> None:
    value = canonicalize_linear_gradient(
        {
            "kind": "linear",
            "colors": ["#123abc", "#abcdef"],
            "stops": [0, 1],
            "start": [0.123456789, 0.9],
            "end": [0.876543219, 0.1],
            "coordinate_space": "glyph_bbox_normalized",
        }
    )
    assert value["colors"] == ["#123ABC", "#ABCDEF"]
    assert value["start"] == [0.123457, 0.9]
    assert value["end"] == [0.876543, 0.1]


@pytest.mark.parametrize(
    "value",
    [None, [], ["#000000"], ["bad", "#FFFFFF"],
     {"colors": ["#000000", "#FFFFFF"], "start": [0, 0], "end": [0, 0]}],
)
def test_invalid_or_degenerate_gradient_abstains(value: object) -> None:
    assert canonicalize_linear_gradient(value) is None
```

**Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest pipeline/tests/test_gradient_model.py -q
```

Expected: collection fails because `typesetter.gradient_model` does not exist.

**Step 3: Implement canonicalization**

Create `gradient_model.py` with:

```python
LINEAR_GRADIENT_SPACE = "glyph_bbox_normalized"


def canonicalize_linear_gradient(value: object) -> dict[str, object] | None:
    """Return a deterministic two-stop normalized linear-gradient value."""
```

Implementation requirements:

- accept a two-item legacy sequence and the structured mapping;
- require two valid `#RRGGBB` colors and uppercase them;
- require finite two-number start/end points;
- reject a start/end distance smaller than `1e-6`;
- clamp normalized points to `[0.0, 1.0]`;
- emit fixed stops `[0.0, 1.0]` for this two-stop increment;
- round point coordinates to six decimals;
- deep-create the result so callers cannot mutate source evidence.

**Step 4: Add failing projection and paint tests**

Test `gradient_parameter_map(mask, spec)` and
`render_linear_gradient_rgb(mask, spec)` with a rectangular core mask:

- vertical red→blue: top core pixels red and bottom core pixels blue;
- horizontal red→yellow: left core pixels red and right core pixels yellow;
- diagonal blue→green: the two projected tails match their declared colors;
- outside-mask pixels stay zero;
- a structured reversed axis reverses the rendered progression.

Use a channel tolerance of at most 2 for exact synthetic endpoints.

**Step 5: Run tests and verify RED**

Run the same focused command. Expected: canonicalization passes; projection and
paint tests fail because the functions are missing.

**Step 6: Implement projection and paint**

Implement:

```python
def gradient_parameter_map(
    mask: np.ndarray,
    gradient: object,
) -> np.ndarray:
    """Evaluate one block-level normalized field over the mask's glyph bbox."""


def render_linear_gradient_rgb(
    mask: np.ndarray,
    gradient: object,
) -> np.ndarray:
    """Return RGB pixels for the canonical field, zero outside the mask."""
```

Compute the glyph-union bounding box once. Normalize all pixel centers against
that box, project them onto `start→end`, divide by squared axis length, clamp to
`[0, 1]`, and interpolate in float RGB before rounding to `uint8`.

**Step 7: Run tests and verify GREEN**

Run:

```powershell
python -m pytest pipeline/tests/test_gradient_model.py -q
```

Expected: all Task 1 tests pass.

**Step 8: Commit only Task 1**

```powershell
git add -- pipeline/typesetter/gradient_model.py pipeline/tests/test_gradient_model.py
git commit -m "feat: add canonical directional gradient model"
```

### Task 2: Detect robust two-dimensional gradients from glyph interiors

**Files:**
- Modify: `pipeline/typesetter/gradient_model.py`
- Modify: `pipeline/typesetter/style_extractor.py:119-376`
- Modify: `pipeline/tests/test_gradient_model.py`
- Modify: `pipeline/tests/test_style_extractor.py`

**Step 1: Add a reusable synthetic source fixture**

In `test_gradient_model.py`, build an uneven five-line glyph-like mask whose line
widths differ. Paint the canonical field through the mask, optionally dilate a
white outline around it, and optionally add a shifted dark shadow. Return the
RGB source and authoritative core mask.

The fixture must not call the detector to create its expected values.

**Step 2: Write failing detector tests**

Parameterize at least these cases:

| Case | Start | End | Colors |
|---|---:|---:|---|
| vertical | `[0.5, 0.0]` | `[0.5, 1.0]` | purple→black |
| horizontal | `[0.0, 0.5]` | `[1.0, 0.5]` | red→yellow |
| diagonal | `[0.0, 0.0]` | `[1.0, 1.0]` | blue→green |
| reverse diagonal | `[0.0, 0.0]` | `[1.0, 1.0]` | cream→navy |
| opposite diagonal | `[1.0, 0.0]` | `[0.0, 1.0]` | magenta→cyan |

Assert:

- status is observed;
- confidence is at least `0.60`;
- cosine similarity between expected and detected axes is at least `0.93`,
  allowing paired axis/color reversal;
- each detected endpoint is within RGB Euclidean distance 24 of its expected
  endpoint;
- the result has the canonical structured shape.

Add negative tests proving solid black, sparse support, and white-outline-only
variation abstain. Add a contamination test with white outline and dark shadow
that still recovers the fill direction/colors.

**Step 3: Run detector tests and verify RED**

Run:

```powershell
python -m pytest pipeline/tests/test_gradient_model.py -k "detect" -q
```

Expected: failures because `detect_linear_gradient` is missing.

**Step 4: Implement robust spatial fitting**

Add:

```python
@dataclass(frozen=True)
class GradientDetection:
    value: dict[str, object] | None
    confidence: float
    reason: str
    metrics: dict[str, object]


def detect_linear_gradient(
    image_rgb: np.ndarray,
    sampling_mask: np.ndarray,
) -> GradientDetection:
    """Fit a supported rank-one linear color field to glyph-interior samples."""
```

Algorithm:

1. Validate matching image/mask shapes and require at least 24 pixels.
2. Normalize sample X/Y positions to the complete mask bounding box.
3. Partition support into a 6×6 spatial grid and retain cells with at least four
   samples; use median position and median RGB per cell.
4. Require at least six supported cells spanning at least 35% of one axis and
   20% of the other.
5. Fit `RGB = intercept + [x, y] @ coefficients` with least squares.
6. Use SVD of the 2×3 coefficient matrix to obtain the dominant spatial axis and
   rank-one explained energy.
7. Orient deterministically: the largest-magnitude axis component must be
   positive. This preserves reversed colors rather than changing axis hashes.
8. Project raw supported pixels onto the axis. Estimate endpoint colors from the
   median of the lowest and highest 12.5% projection tails.
9. Derive start/end points from those supported projection quantiles through the
   support centroid; clamp to normalized bounds.
10. Compute confidence from endpoint distance, spatial coverage, rank-one energy,
    tail support, and median residual. Record each term in metrics.
11. Abstain below 24 RGB-distance units, below 0.60 confidence, below 0.72
    rank-one explained energy, or when residual exceeds the supported endpoint
    separation.

Do not hardcode purple, black, band IDs, chapter IDs, or expected axis angles.

**Step 5: Integrate the V2 owner extractor with a failing test**

In `test_style_extractor.py`, call `extract_text_style_evidence_v2()` with the
synthetic diagonal source and authoritative mask. Assert that
`attributes['gradient'].value` is structured and that its confidence/direction
match the detector result. Assert that `metrics` contains gradient fit support,
residual, and direction diagnostics.

Run the single test and verify that the old top/bottom list fails it.

**Step 6: Replace the median-Y detector**

In `extract_text_style_evidence_v2()`, replace the block that divides samples at
the median Y with `detect_linear_gradient(image_rgb, fill_sampling_mask)`. Emit
`_v2_observed(detection.value, detection.confidence)` only when a value exists;
otherwise emit `_v2_unknown(detection.reason)`. Copy diagnostic metrics using a
`gradient_` prefix.

Keep coarse owner masks abstaining before fitting.

**Step 7: Run extractor and model tests**

```powershell
python -m pytest pipeline/tests/test_gradient_model.py pipeline/tests/test_style_extractor.py -q
```

Expected: all pass.

**Step 8: Commit only Task 2**

```powershell
git add -- pipeline/typesetter/gradient_model.py pipeline/typesetter/style_extractor.py pipeline/tests/test_gradient_model.py pipeline/tests/test_style_extractor.py
git commit -m "feat: detect source gradient direction and colors"
```

### Task 3: Carry the structured gradient through Style V2 and owner profiles

**Files:**
- Modify: `pipeline/typesetter/style_contract.py:98-273`
- Modify: `pipeline/typesetter/style_policy.py:218-383`
- Modify: `pipeline/typesetter/owner_style.py:180-240`
- Modify: `pipeline/tests/test_style_contract.py`
- Modify: `pipeline/tests/test_style_policy_v2.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`
- Modify: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Write failing contract and policy tests**

Test that:

- V1 two-color evidence migrates to the canonical vertical structured value;
- V2 `to_dict → from_dict → sha256` preserves a diagonal value exactly;
- policy accepts a confident valid structured value;
- policy abstains from invalid/degenerate structured values;
- a structured gradient supersedes solid fill just as the legacy list did;
- `normalize_auto_typesetting_style` deep-copies the structured gradient;
- `_has_authenticated_source_gradient` accepts structured and legacy valid
  values, and rejects same-color/invalid values.

Run focused tests and verify failures against the list-only logic.

**Step 2: Canonicalize at contract boundaries**

Use `canonicalize_linear_gradient()` in `style_evidence_v2_from_v1()` and while
reading a V2 `gradient` attribute in `style_evidence_v2_from_dict()`. Do not alter
other attribute values.

In `evaluate_style_attribute()`, require successful gradient canonicalization
before applying the existing gradient confidence threshold. Return the canonical
mapping as the approved value.

**Step 3: Materialize owner style without flattening direction**

Update owner-style construction so:

```python
gradient = canonicalize_linear_gradient(applied.get("gradient"))
if gradient is not None:
    style["cor_gradiente"] = gradient
    style["cor"] = gradient["colors"][0]
```

Update authenticated-gradient checks and automatic-style normalization to
preserve the mapping via deep copy. No field may convert it back to a list.

**Step 4: Run focused tests and verify GREEN**

```powershell
python -m pytest pipeline/tests/test_style_contract.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_typesetting_style_policy.py -q
```

Expected: all pass.

**Step 5: Commit only Task 3**

```powershell
git add -- pipeline/typesetter/style_contract.py pipeline/typesetter/style_policy.py pipeline/typesetter/owner_style.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_typesetting_style_policy.py
git commit -m "feat: preserve directional gradients in style contracts"
```

### Task 4: Rasterize and observe the field along its real axis

**Files:**
- Modify: `pipeline/typesetter/glyph_rasterizer.py:331-410,477-705`
- Modify: `pipeline/tests/test_glyph_rasterizer.py`
- Modify: `pipeline/typesetter/style_materialization.py:796-920`
- Modify: `pipeline/tests/test_style_materialization.py`

**Step 1: Write failing raster tests**

Add structured horizontal and diagonal gradient tests to
`test_glyph_rasterizer.py`. Use a multiline core union. Assert endpoint colors
at pixels near the projected tails and assert that equal normalized positions on
different lines receive equal colors. This proves the field is block-level.

Add a legacy list test proving it still paints vertically.

Run the focused tests and verify that the current implementation always paints
top-to-bottom.

**Step 2: Render with the shared field evaluator**

In `rasterize_v2_glyph_layers()`:

- canonicalize `style['gradient']`;
- call `render_linear_gradient_rgb(core, gradient)`;
- compose the returned RGB only through `core` alpha;
- store the complete canonical mapping in `applied['gradient']` and layer
  evidence;
- retain solid fill fallback when canonicalization returns `None`.

Delete the row-by-row vertical interpolation from this path.

**Step 3: Write failing directional observation tests**

Extend materialization tests so observation of a horizontal/diagonal raster:

- reports the same canonical gradient value;
- detects a deliberately reversed or rotated output as a mismatch;
- passes an equivalent legacy vertical input after canonicalization.

**Step 4: Sample observed colors along the declared axis**

Change `_gradient_endpoints()` to accept the expected canonical gradient. Use
`gradient_parameter_map()` to select the lowest/highest 12.5% supported core
tails and return their median colors. Observation must include the canonical
axis and colors, not only vertically sampled endpoints.

Ensure `build_materialization_observation()` canonicalizes expected and observed
gradient values before equality/comparison and emits a precise direction or
endpoint mismatch.

**Step 5: Run focused tests and verify GREEN**

```powershell
python -m pytest pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_style_materialization.py -q
```

Expected: all pass.

**Step 6: Commit only Task 4**

```powershell
git add -- pipeline/typesetter/glyph_rasterizer.py pipeline/typesetter/style_materialization.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_style_materialization.py
git commit -m "feat: render and audit directional text gradients"
```

### Task 5: Unify all renderer paths on block-level gradient evaluation

**Files:**
- Modify: `pipeline/typesetter/renderer.py:1440-1750,2413-2460,16639-16650,17890-17960,18349-18395`
- Modify: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing renderer-path tests**

Add tests for:

1. V2 owner rendering with a diagonal field and uneven translated line lengths.
2. Safe TextPath rendering with a horizontal field.
3. Legacy rendering with a vertical two-color list.
4. A no-line-restart sentinel: pixels with the same normalized projection on
   separate lines must have the same color within tolerance.
5. A field where the second translated line is much shorter than the first;
   the short line samples its actual block position rather than receiving both
   endpoint colors locally.

Run each new test individually and verify RED.

**Step 2: Canonicalize once at renderer boundaries**

Add a private helper that calls `canonicalize_linear_gradient()` for
`plan['cor_gradiente']`. Update all checks that currently assume a list or index
`gradient[0]`/`gradient[1]` to use `gradient['colors']`.

In `_render_v2_owner_text_layer()`, pass the complete canonical mapping to the
glyph rasterizer and sealed materialization plan. Do not rebuild a two-color list.

**Step 3: Replace duplicate vertical loops**

Refactor `_apply_safe_gradient_text()` and `_apply_gradient_text()` to build one
mask representing the complete block, evaluate `render_linear_gradient_rgb()`
once for that union, and then composite it. Preserve existing outline ordering.

If a legacy caller provides two color arguments, canonicalize them through the
vertical compatibility adapter before rendering.

**Step 4: Update contrast guards**

Any readability/contrast code that checks gradient endpoints must read the
canonical `colors` array. It may disable an unsafe gradient as before, but must
not change its axis or replace one endpoint silently.

**Step 5: Run focused renderer tests**

```powershell
python -m pytest pipeline/tests/test_typesetting_renderer.py -k "gradient or materialization" -q
```

Expected: all selected tests pass.

**Step 6: Run adjacent typesetting suites**

```powershell
python -m pytest pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_typesetting_style_policy.py pipeline/tests/test_typesetting_renderer.py -q
```

Expected: all pass.

**Step 7: Commit only Task 5**

```powershell
git add -- pipeline/typesetter/renderer.py pipeline/tests/test_typesetting_renderer.py
git commit -m "feat: apply gradients across complete translated text blocks"
```

### Task 6: Update QA and debug consumers without losing direction

**Files:**
- Modify: `pipeline/debug_tools/style_audit_report.py`
- Modify: `pipeline/debug_tools/style_copy_score.py`
- Modify: `pipeline/qa/style_fidelity.py`
- Modify: `pipeline/tests/test_style_audit_report.py`
- Modify: `pipeline/tests/test_style_copy_score.py`
- Modify: the owning QA test file identified by `rg` for `style_fidelity`

**Step 1: Inventory list-only consumers**

Run:

```powershell
rg -n "gradient_colors|cor_gradiente|len\(gradient|gradient\[[01]\]" pipeline -g "*.py"
```

Classify each hit as legacy input, canonical consumer, presentation-only report,
or unrelated background gradient. Record the classification in the task notes;
do not mechanically replace unrelated image-gradient code.

**Step 2: Write failing QA/report tests**

Assert that reports expose both `colors` and `direction`, score a correctly
materialized diagonal field as applied, and keep reading legacy two-color rows.
Assert that a direction mismatch cannot receive a perfect gradient-fidelity
score merely because endpoint colors match.

Run focused tests and verify RED.

**Step 3: Implement compatibility adapters**

Canonicalize style gradients at report/QA input boundaries. Keep old exported
`gradient_colors` arrays where external report schemas require them, but add the
canonical direction and ensure scoring uses it. Do not change unrelated report
fields.

**Step 4: Prove no unsafe list-only consumer remains**

Repeat the inventory command. Every remaining indexed access must either be in
the canonical helper or explicitly handle both canonical and legacy values.

**Step 5: Run focused tests and verify GREEN**

```powershell
python -m pytest pipeline/tests/test_style_audit_report.py pipeline/tests/test_style_copy_score.py pipeline/tests -k "style_fidelity" -q
```

Expected: selected tests pass.

**Step 6: Commit only Task 6**

Stage only the files actually changed by this task and commit:

```powershell
git commit -m "fix: audit directional gradient fidelity"
```

### Task 7: Add a systemic synthetic gradient matrix and visual contact sheet

**Files:**
- Create: `pipeline/tests/test_directional_gradient_pipeline.py`
- Create: `pipeline/debug_tools/render_directional_gradient_matrix.py`
- Create or modify: the narrow test for this debug tool under `pipeline/tests/`

**Step 1: Write failing end-to-end synthetic tests**

For each matrix case, execute:

```text
synthetic source → authoritative interior mask → V2 extraction
→ style decision → owner-style mapping → glyph rasterizer → observation
```

Use at least:

- purple→black diagonal;
- red→yellow horizontal;
- blue→green diagonal;
- white→cyan vertical;
- dark→light reverse vertical;
- magenta→orange opposite diagonal;
- solid black negative control;
- saturated fill with white outline and shadow contamination.

Use deliberately different source and translated multiline masks. Assert no
source line geometry leaks into the translated mask, the direction cosine and
endpoint colors stay within tolerance, and materialization status is `match`.

Run and verify RED at the first unimplemented integration seam.

**Step 2: Implement only the missing integration adapters**

Do not add special cases to the detector. Fix the owning general contract or
adapter revealed by the failing case, rerunning the single case after each fix.

**Step 3: Implement the visual matrix tool**

The tool must deterministically render a contact sheet containing, per row:

- synthetic source;
- authoritative glyph mask;
- detected colors/direction/confidence;
- translated raster;
- materialization result.

It accepts `--output-dir`; default output must be outside tracked source. It must
not need OCR, translation, network access, or GPU.

**Step 4: Test the tool**

Test deterministic case IDs, JSON evidence, image dimensions, and that the
declared output directory is the only write location.

**Step 5: Run the matrix and inspect it visually**

```powershell
python pipeline/debug_tools/render_directional_gradient_matrix.py --output-dir .codex-tmp/directional-gradient-matrix
```

Open the generated contact sheet at original resolution. Record GO/NO-GO per
case in its generated JSON. Any line-local restart, wrong direction, contaminated
endpoint, or false positive is NO-GO.

**Step 6: Run tests and verify GREEN**

```powershell
python -m pytest pipeline/tests/test_directional_gradient_pipeline.py pipeline/tests -k "directional_gradient_matrix" -q
```

Expected: all pass.

**Step 7: Commit only Task 7 source/tests**

Do not add `.codex-tmp` artifacts.

```powershell
git add -- pipeline/tests/test_directional_gradient_pipeline.py pipeline/debug_tools/render_directional_gradient_matrix.py
git commit -m "test: validate directional gradient matrix"
```

### Task 8: Run the real chapter 39 band 006 through the automatic pipeline

**Files:**
- Read: `.codex-tmp/mch39_006_cc_auto_20260803/config.json`
- Create outside Git: `.codex-tmp/mch39_006_directional_gradient_20260803/config.json`
- Inspect: generated stages `08_inpaint`, `09_typeset`,
  `10_copyback_reassemble`, and `11_qa_export_gate`

**Step 1: Freeze the pre-run state**

Record:

```powershell
git status --short -- fonts pipeline/typesetter pipeline/tests pipeline/debug_tools docs/plans
git diff -- pipeline/typesetter pipeline/tests pipeline/debug_tools
```

Copy the existing 006 run configuration into the new `.codex-tmp` directory and
change only run/output paths. Preserve the exact source image and automatic OCR,
style, font, layout, inpaint, and QA settings. Do not inject expected colors,
direction, font, owner ID, or band ID.

**Step 2: Run the real pipeline**

Run the same real entrypoint used by the prior 006 validation:

```powershell
python pipeline/main.py .codex-tmp/mch39_006_directional_gradient_20260803/config.json
```

Confirm logs report CUDA/GPU execution for GPU-capable stages. A CPU-only style
fit is acceptable because it is small NumPy work; the full visual pipeline must
not silently fall back from configured GPU inference.

**Step 3: Validate structured evidence**

Inspect the generated project/evidence and assert:

- gradient value is structured, not a list;
- direction has material X and Y components for band 006;
- the purple endpoint is materially brighter/more saturated than the old
  `#170A3B` result;
- no source- or band-specific override exists;
- automatic font selection remains `CCTotallyAwesome W00 Bold.ttf` unless the
  unchanged matcher independently abstains;
- materialization observation matches colors and direction.

**Step 4: Inspect visual stages at original resolution**

Inspect:

- original/source glyphs;
- final inpaint before text;
- typeset layer;
- copyback band;
- final translated page;
- export gate and owner invariant report.

Acceptance for band 006:

- upper-left PT-BR glyphs visibly retain the source purple family;
- the field progresses toward black to the right and bottom;
- the field is continuous across all PT-BR lines;
- outline stays separate and does not tint the fill;
- text remains centered, uppercase, non-pixelated, and fully inside the balloon;
- inpaint does not regress;
- export gate is PASS and the style subgate does not report a high-confidence
  directional mismatch.

**Step 5: Generate and inspect the real comparison**

Create a contact sheet outside Git with original, previous translated result,
new translated result, source gradient evidence, and rendered gradient evidence.
Open it at original resolution and record a candid GO/NO-GO. Exit code zero alone
is not success.

**Step 6: If NO-GO, return to the first failing general layer**

Classify the failure as mask sampling, field fit, contract transport, layout
projection, rasterization, copyback, or QA observation. Add a general failing
test that reproduces the class before changing production code. Never patch band
006 directly.

### Task 9: Regression verification and handoff

**Files:**
- Verify all modified files
- Update the implementation report only if the repository's current plan requires
  one; do not edit unrelated historical reports

**Step 1: Run the focused directional suite**

```powershell
python -m pytest pipeline/tests/test_gradient_model.py pipeline/tests/test_style_extractor.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_style_materialization.py pipeline/tests/test_directional_gradient_pipeline.py -q
```

Expected: all pass.

**Step 2: Run the adjacent typesetting/QA suite**

```powershell
python -m pytest pipeline/tests/test_typesetting_style_policy.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_style_audit_report.py pipeline/tests/test_style_copy_score.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_final_pixel_export_gate.py -q
```

Expected: all pass.

**Step 3: Run static checks**

```powershell
python -m compileall -q pipeline/typesetter pipeline/debug_tools
git diff --check
```

Expected: no errors.

**Step 4: Audit the final dirty tree safely**

```powershell
git status --short
git diff --stat
git diff -- pipeline/typesetter pipeline/tests pipeline/debug_tools docs/plans
```

Separate pre-existing changes from this plan's commits. Do not clean either set.
Verify `.codex-tmp` was not staged.

**Step 5: Final visual verdict**

Report separately:

- synthetic matrix test result;
- synthetic visual GO/NO-GO;
- real 006 pipeline result and runtime;
- real 006 visual GO/NO-GO;
- detected colors, start/end, confidence, and materialization status;
- export gate status;
- every remaining blocker or known limitation.

Do not describe the plan as complete if the real output remains too dark, restarts
per line, uses the wrong axis, regresses inpaint/layout, or has a blocked export
gate.

