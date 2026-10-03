# Owner-level preparation parallel experiment

User approved testing concurrency within style/font/evidence preparation. No production integration. Preserve source-page input, graph context, all owner outputs and exceptions, and stable merge order.

1. Test deterministic owner-map and exception propagation RED/GREEN.
2. Experimental wrappers around initial glyph capture, protected-art capture and owner style/font capture. Two/four/eight internal threads versus one. Separate mutable font-matcher cache per thread; page glyph cache synchronized to preserve correctness. Phase dependency barriers retained. No nested font-candidate pool: font analyses run concurrently as part of independent owners.
3. Fresh page1 controls1/2/4; compare full style captures and glyph/protected arrays plus mask fixtures. Reject any differing output.
4. If valid, measure all18pages with controlled outer/inner combinations (e.g.8x1,8x2,4x4); compare64 mask inputs and all page phase receipts. Record RAM and wall-time rather than multiplying theoretical workers.
5. Record whether more inner threads help; retain slower/failed evidence. Tests and report, no commits/resets or modification of old runs.
