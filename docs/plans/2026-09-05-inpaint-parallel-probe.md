# Mask and inpaint profiling experiment

User approved isolated tests of model/copies/hashes/verification/artifacts plus parallelism. No production integration or removal of safety checks. Preserve dirty checkout and prior outputs. Use the existing Chapter24 source pixels and graph/translation artifacts, never retrigger Google, Ollama or OCR.

1. Capture actual build_owner_mask_plan inputs from execute_owner_page_graph with historical translation override; intercept before rendering/inpaint and persist compressed private experiment fixtures. Verify canonical page pixel hashes. Capture failures explicitly; do not claim historical equality without checking.
2. Replay real mask builder and real owner inpaint, profiling nested spans exclusively with thread-local stacks: masks, engine, hashes, residual and other wrapper work. Retain output pixel and mask hashes, outside/protected changed counts and errors. Separate setup, stage wall and accumulated CPU time.
3. Compare 1, 2 and 4 bounded CPU workers with serialized GPU access (CPU/GPU overlap). Test two independent engines concurrently only if memory and engine behavior allow. Reuse decoded immutable page inputs within a run; don't call it E2E/cold boot. Compare full output hashes and flags before any speedup claim.
4. Optional debug-image overhead isolated from required mask/proof checks. Model-only times must not be presented as entire inpaint. A profile with rejected owners still records all cases, not only successes.
5. Focused tests for timer accounting and immutable hash reuse; frozen result equality for real replay, sample visual check when outputs change; repeated baseline/candidate. Final report records actual scope, limits, all errors and whether each proposed experiment completed.

Files only under artifacts/e2e/container_parallel_20260905/inpaint_probe*. No commits, resets, worktrees or production edits.
