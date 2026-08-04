# Style Copy Directional Gradient Design

**Date:** 2026-08-03

**Status:** Approved

## Problem

Style Copy V2 currently reduces every detected gradient to two colors ordered by
vertical position. The authoritative extractor divides glyph pixels around the
median Y coordinate, while the glyph rasterizer interpolates those colors only
from the top to the bottom of the translated glyph bounding box. This loses the
source direction, biases endpoint colors toward the middle of the transition,
and cannot reproduce horizontal or diagonal gradients.

The chapter 39 band 006 exposes all three defects. Its source fill changes in
both X and Y: the upper-left glyph interiors are saturated purple, the lower and
rightward glyph interiors approach black, and the progression remains continuous
across the whole text block. The current evidence records only
`["#170A3B", "#000000"]`, then paints a vertical gradient. The translated result
is consequently too dark and has the wrong spatial progression.

## Goal

Detect a robust linear color field from authoritative source glyph interiors and
materialize that field continuously over the complete translated glyph body,
preserving endpoint colors and direction across different line breaks and text
lengths.

## Non-goals

- Copying source letter pixels or rasterizing the English text as a texture.
- Restarting a gradient for every rendered line.
- Guessing a gradient when the source supports only a solid fill.
- Supporting arbitrary illustrations, patterned fills, or nonlinear rainbow
  textures in this increment.
- Changing font selection, translation, ownership, inpaint, or functional layout
  policy.

## Considered approaches

### 1. Normalized two-dimensional linear field — selected

Fit a spatial axis to the color variation inside the authoritative glyph mask,
store the axis and robust endpoint colors in normalized glyph-block coordinates,
and evaluate the field for every translated glyph pixel. This survives reflow,
does not depend on source letter shapes, and supports vertical, horizontal,
diagonal, and reversed gradients with the same contract.

### 2. Source color texture transfer

Warping or resampling the source fill map could reproduce local irregularities,
but it binds color to the English glyph geometry. Different PT-BR line breaks
would stretch holes and letter-specific artifacts into the new text. It also
risks copying antialiasing and outline pixels. This is rejected.

### 3. Independent per-line gradients

Applying a separate gradient to every translated line is easy to implement, but
it visibly repeats the starting color and destroys the continuous block-level
progression. It contradicts the approved visual criterion and is rejected.

## Gradient contract

The canonical Style V2 gradient value becomes a structured value:

```json
{
  "kind": "linear",
  "colors": ["#4A2497", "#08080A"],
  "stops": [0.0, 1.0],
  "start": [0.0, 0.0],
  "end": [1.0, 1.0],
  "coordinate_space": "glyph_bbox_normalized"
}
```

`start` and `end` describe the detected direction in the normalized bounding box
of the complete glyph union, not an individual line. Colors are paired with the
ordered endpoints. A reversed source therefore reverses the colors rather than
silently normalizing the direction.

Legacy two-color lists remain readable. They are canonicalized as a vertical
top-to-bottom field with `start=[0.5, 0.0]` and `end=[0.5, 1.0]`. Newly captured
evidence always emits the structured form. This preserves existing projects and
tests while preventing further information loss.

The value remains one Style V2 `gradient` attribute. Direction is not split into
a second independently gated attribute, because colors without their spatial
mapping do not represent the observed effect.

## Detection

Detection continues to use the owner-bound, pre-inpaint glyph mask. It must not
redetect text from pixels. The existing distance-to-edge interior mask remains
the first defense against outlines and antialiasing.

The detector will:

1. Normalize authoritative glyph coordinates to the full glyph-union bounding
   box.
2. Divide the support into spatial cells and compute robust cell colors from
   glyph-interior pixels. Cells with insufficient support are discarded.
3. Fit a two-dimensional linear RGB color plane to the supported cells using a
   robust residual pass.
4. Derive the spatial direction from the fitted color-plane gradient rather
   than from the aspect ratio or orientation of the text itself.
5. Project supported samples onto that direction.
6. Estimate endpoint colors from robust medians near the supported projection
   tails, avoiding unconstrained extrapolation beyond observed glyph pixels.
7. Calculate confidence from endpoint color distance, spatial coverage,
   explained variation, tail support, and fit residual.

The detector abstains when support is sparse, endpoint colors are too close, the
spatial fit does not explain the color variation, or likely background/outline
contamination dominates. A solid fill therefore remains a solid fill. White
outlines and dark shadows must not become gradient endpoints.

The fit is deliberately limited to a two-stop linear gradient. The structured
contract contains a `stops` array so a later calibrated multi-stop detector can
extend it without changing consumers.

## Materialization

The glyph compositor receives the canonical structured gradient. For every pixel
in the complete translated core mask it:

1. maps the pixel to normalized coordinates in the bounding box of the complete
   glyph union;
2. projects the coordinate onto the `start→end` axis;
3. clamps the resulting parameter to the declared stop range;
4. interpolates colors; and
5. paints only through the core glyph alpha.

This means different translated line lengths sample different parts of the same
field naturally, while the progression never restarts per line. Stroke, shadow,
and glow remain separate layers and cannot influence fill interpolation.

The older safe and legacy render paths will call the same canonical gradient
helper instead of maintaining their own vertical loops. This avoids a result
changing when a text block switches renderer path.

## Evidence and audit

Materialization evidence must report the canonical structured gradient, including
direction and endpoint colors. Observation samples the rendered core near the
expected projected tails, not merely the top and bottom rows. A direction change
therefore becomes an auditable mismatch instead of passing because two colors
were present somewhere in the output.

Canonical hashing and comparison use rounded normalized coordinates and uppercase
hex colors. Equivalent legacy vertical values canonicalize to the same structured
representation before comparison.

## Error handling and fallback

- Invalid or degenerate axes abstain with `invalid_gradient_axis`.
- Insufficient interior support abstains with
  `insufficient_gradient_spatial_support`.
- Weak color separation abstains with `solid_fill_no_gradient`.
- High residual or low explained variation abstains with
  `nonlinear_or_contaminated_gradient`.
- Invalid legacy values do not crash rendering; they fall back to the approved
  solid fill and remain visible in Style V2 abstention evidence.
- No owner identity, translation payload, layout geometry, or inpaint decision is
  changed by gradient handling.

## Validation

TDD coverage will include:

- vertical purple→black;
- horizontal red→yellow;
- diagonal blue→green;
- reversed light→dark and dark→light axes;
- a 006-like diagonal purple→black field over uneven multiline glyph masks;
- different translated line lengths with one continuous block-level field;
- solid text negative controls;
- white-outline, shadow, antialias, sparse-mask, and background-contamination
  negative controls;
- legacy two-color list compatibility;
- Style V2 decision, materialization, observation, and hash round trips.

After focused and regression tests, the real GPU pipeline will run only chapter
39 band 006. Validation will inspect the source, captured evidence, inpaint,
typeset, copyback, and export gate. A contact sheet will compare the English
source, PT-BR result, and synthetic direction/color cases at native scale.

## Acceptance criteria

- Band 006 detects a diagonal purple→black field with a purple endpoint visibly
  closer to the source than `#170A3B` and renders it continuously over the PT-BR
  block.
- The result does not restart purple on each translated line.
- Horizontal, vertical, diagonal, reversed, and different-color synthetic cases
  recover direction and endpoint colors within declared tolerances.
- Solid text and contaminated negative controls do not acquire false gradients.
- Existing two-color projects still render as vertical gradients.
- Style materialization evidence verifies both color and direction.
- The real band 006 run uses the automatic pipeline decision; no owner- or
  band-specific override is added.

