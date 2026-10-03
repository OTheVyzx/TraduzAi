# Container Sweep Integrated Experiment Implementation Plan

**Goal:** Measure full preparation with page sweep, preserving outputs and owner execution order.

**Architecture:** Lazily resolve a superset of semantic component-union boxes on the first real container query. Keep only small selected results; do not suspend owners, retain their masks, or reorder graph mutations. Existing geometry decisions, fallback and evidence remain unchanged. Instrument unexpected boxes with conservative original fallback; any fallback blocks equivalence-based promotion.

**Tech Stack:** Python, NumPy, OpenCV, existing experimental preparation harness.

1. Add isolated page sweep wrapper and tests: component-union geometry, exact requests, unexpected-query fallback, one batch per page.
2. Record complete preparation executor result and input graph after execution in both modes, alongside existing mask fixtures and style receipts.
3. Run fresh full preparation in ABBA order (16 processes, same flags, no competing heavy work), source hashes frozen.
4. Compare 64 mask inputs, style receipts, complete executor output, graph, sequence of queries and terminal states. Keep capture-only rejections distinct from real mask-builder safety rejection.
5. Report total external time, RSS and internal sweep time separately. No production integration, commits, training, OCR or translation. Real cleanup rejection requires separate replay evidence, not inference from capture-only outputs.
