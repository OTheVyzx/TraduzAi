# Google batches experiment design and plan

Goal: test the user-approved indexed batching design against Chapter24, seeking <=5 seconds for the translation stage, without Ollama or cross-run translation cache.

Scope: experimental files only in artifacts/e2e/container_parallel_20260905. Preserve all existing production modifications, runs and source files; no commits, worktree moves or automatic integration.

Architecture: preserve each existing owner (including its component membership) as the translation unit. Capture the exact preprocessed/protected strings through the real Google preparation path, send indexed groups through Google, parse exact unique markers, then run the existing postprocessing and per-owner language validator. Do not invent a new balloon geometry or split a joined speech by translated character count. Raw responses and per-request timings remain in each fresh output directory. Preserve rejected records; do not satisfy the target with skipped work.

Alternatives: (1) largest bounded sequential batches reduce request count; (2) smaller batches with 2-4 workers overlap network waits; (3) use the Google public JSON transport already present as a fallback in production, without the preceding HTML attempts. Compare measured accuracy and timing; do not attribute cold/warm server effects to local changes. Bound network concurrency at four, enforce timeouts and stop on rate limiting rather than evading it.

Tasks:
1. Add parser/packer tests in test_google_batch_probe.py: roundtrip, missing/duplicate/reordered IDs, wrong count, oversized item, preserved multi-line unit. Run RED then implement google_batch_probe.py and run GREEN.
2. Load historical 116 source requests and verify source hashes, freeze code hashes and reference outputs. Preserve 42 language-policy cases through the current validator. Include capture, batching, request waits, retry, postprocessing, validation and artifact writes in the measured stage; report process startup separately.
3. Run bounded batches then 2-4 concurrent batches, with fresh local cache for each process. Retry only rejected units once, and split corrupt batches into smaller groups within a bounded retry policy. Record HTTP counts, decoded response bytes and timings separately from logical owner attempts.
4. Compare all 116 IDs, accepted/unbound sets, target differences and validator reasons. Inspect differences; equality/validator success alone is not human translation-quality approval.
5. If a candidate reaches <=5s with no acceptance regression, repeat it at least twice with new output paths and fresh caches. Otherwise report the best valid result, bottleneck and limits honestly; no artificial cache hits or unbounded service load.

Acceptance: zero Ollama requests; all owners accounted for; no missing/duplicate/reordered marker accepted; no accepted historical owner becomes unbound; no lost owner identity; best and all repeat times reported. No E2E export claim. A five-second measurement is not a maximum-latency guarantee for an external service.
