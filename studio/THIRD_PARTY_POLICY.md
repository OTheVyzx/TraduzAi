# Third-Party Reuse Policy

This policy keeps TraduzAI Studio deliberate about reuse.

## Allowed

- Current TraduzAI source owned by this project.
- Published npm dependencies under MIT, Apache-2.0, ISC, BSD, or compatible
  licenses, with notices.
- Koharu code only when Studio is distributed as GPL-3.0-only or another
  GPL-compatible license.

## Inspiration Only

- Koharu UI architecture may guide the Studio brush/canvas design, but copying
  coupled UI modules should wait until the Studio scene and backend contracts
  are stable.
- No-license detector repositories may be studied for behavior and benchmarks,
  but implementations must be clean-room.
- PSD export starts from TraduzAI-owned code. External PSD code can be studied
  for behavior, but copied implementation must pass license review first.

## Blocked Unless Relicensed Or Permission Is Granted

- `manga-cleaner-3.0.0` source, UI, assets, models, and binaries.
- `MangaCleaner_GPU` packaged source, assets, models, and binaries.
- Root `NotAnotherBubbleCleaner` app/notebook/utils code.
- `Bubble-Detector-YOLOv4` code, models, weights, and datasets.
- `bubble_segmentation` code, models, weights, and datasets.

## Required Before Copying External Code

1. Identify the exact source path and license file.
2. Add the license text to third-party notices.
3. Add a short note explaining why dependency reuse is better than a clean-room
   implementation.
4. Add tests that exercise the imported behavior through Studio APIs.
