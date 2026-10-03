# Preparation reuse and parallel measurement

Goal: measure fresh Chapter24 preparation, preserve exact inputs and cleanup outputs, test 4/6/8 page workers. User approved the combined experiment; no production integration, commits, resets or old artifact edits.

Architecture: new experimental wrapper around existing capture, page-scoped bounded glyph-raster cache, local connected-component analysis for protected evidence. Keep full-size public masks and all protection decisions. Instrument exclusive function times and fixture serialization; profile one control page before chapter trials. No OCR/translation calls. Existing capture and production stay intact.

Steps:
1. Add failing tests for bounded cache, page isolation, local protected evidence equality including border/contact/provenance/error cases.
2. Implement experimental helper and capture wrapper; run tests.
3. Measure profile control page 1 and candidate page 1; require identical prepared fixtures.
4. Run fresh all-18-page candidates with 8, 6 and 4 workers separately, external profiler and timeouts. Compare all prepared fixtures against the completed full control, not only counts.
5. Replay best candidate with 4 CPU workers, one GPU and residual ROI; compare all 64 results including existing rejection. Report preparation/core separately. Do not infer integrated overlap savings.
6. Save results, breakdown, limitations. If equivalence fails, reject that optimization, preserve evidence and investigate before any speed claim.

Files: artifacts/e2e/container_parallel_20260905/preparation_probe.py, preparation_optimizations.py, test_preparation_optimizations.py; generated outputs in new prep_* directories only.
