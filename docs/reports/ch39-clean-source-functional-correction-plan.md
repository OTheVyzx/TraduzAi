# Chapter 39 clean-source functional correction plan

Status: planning only. No functional pipeline change is included in the corpus-preparation work.

## Goal

Guarantee that each eligible speech-balloon owner is resolved atomically from a clean source, has its English source pixels removed, and receives exactly one verified PT-BR materialization. Historical OCR/repaint output must never become trusted translation provenance merely because it looks like PT-BR.

## Scenario A — real run over the clean English CBZ

The canonical input is the nine-page Chapter 39 CBZ recorded by the private corpus manifest. The future runtime must:

1. discover all source components in full-page coordinates;
2. reconcile components into one stable TextOwner per semantic balloon body;
3. OCR the clean English pixels before any mutation;
4. classify source contamination as clean using pre-translation evidence;
5. obtain a PT-BR target through a verifiable translation authority chain;
6. seal the accepted target before hydration, style, layout, or rendering;
7. remove every source glyph inside an independently bounded mutation envelope;
8. materialize the sealed target exactly once per TextOwner;
9. preserve SFX, credits, domains, outlines, and protected artwork;
10. independently re-open the final pixels and fail on English residual, owner fragmentation, duplicate materialization, or out-of-mask mutation.

The full-page regions corresponding to legacy crops 003 and 005 must produce semantically valid PT-BR from the clean strings “ARE YOU DOING TOTO* AGAIN?” and “SO, ON WHAT DID YOU PLACE YOUR BET?”. They must not reuse the corrupt historical OCR/targets recorded in the private translation-reference ledger.

## Scenario B — contaminated historical crop

The crop is a targeted regression artifact, not a clean chapter page. The future runtime must:

1. capture source pixels and contamination evidence before translation binding;
2. reject `already_target_language_repaint`, source passthrough, target-equals-contaminated-source, and fused OCR as trusted target authority;
3. use an independently approved target only when its provenance is hash-linked to the owner and source candidate;
4. otherwise execute `cleanup_only`, produce zero glyph materializations, and end in REVIEW with a canonical review record;
5. never report a false PASS.

## Implementation order and TDD checkpoints

1. Lock real codecs and acyclic semantic/artifact hashes for requests, attempts, bindings, owner resolution, mutation authorities, outcomes, page evidence, QA, gate, and publication.
2. Add provenance-specific validation to TranslationBinding; caller-provided `trusted=true` is never authority.
3. Add a persisted page-space pre-translation evidence phase before request construction.
4. Run contamination classification in shadow mode against the 18 synthetic fixtures and calibration chapters.
5. Introduce a discriminated resolution result: replace, cleanup-only, preserve, or exclude, with exact owner/component partitions.
6. Add a cleanup-only authority and execution branch before style/layout/typeset; it cannot require a target or glyph authority.
7. Seal replacement targets after an accepted binding and before hydration; prove hydration cannot change the target hash.
8. Enforce one owner → one transaction → one materialization, while allowing multiple OCR components and paragraphs to belong to that owner.
9. Compose only inside independent mutation envelopes and assert unchanged pixels outside them.
10. Re-derive residual language and owner coverage from persisted final pixels.
11. Propagate REVIEW atomically through Python, Rust, TypeScript, UI history, export gate, and publication receipts.
12. Add an offline TranslationReplayLedger and forbid network/provider calls in reproducibility runs.
13. Run regression over clean full pages and contaminated crops separately.
14. Run the sealed holdout only after a separate curator creates expectations.

## Required tests

- Clean full-page source resolves through a trusted target and reaches PASS.
- Contaminated crop rejects target-equals-source and repaint passthrough.
- Multiple OCR boxes and two paragraphs yield one owner and one materialization.
- Two nearby balloons remain two owners.
- Hydration cannot alter a sealed target.
- A second materialization for the same owner is rejected.
- Cleanup-only has no target/glyph materialization and requires REVIEW.
- Physical English residual forces REVIEW/BLOCK according to the canonical gate.
- SFX, credits, domains, balloon outlines, and protected artwork remain unchanged.
- Every pixel outside the allowed mutation mask is byte-identical.

## Acceptance boundary

Do not start functional tuning until the priority masks are manually approved, the corpus validator passes, an offline translation replay ledger exists, and a separate holdout curator has sealed private expectations. A passing process exit code alone is not visual or semantic acceptance.
