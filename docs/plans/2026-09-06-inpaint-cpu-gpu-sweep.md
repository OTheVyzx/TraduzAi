# Inpainting CPU/GPU sweep

User requested GPU concurrency 2 through 8, plus more CPU workers. Experimental only, frozen 64 fixtures from prep_combined_w8_r2/preparation, exact historical output reference. No preparation reruns or production integration.

1. Expand experimental CLI to CPU 1..16/GPU 1..8, RED/GREEN range test. Keep original defaults. Independent AOT models with independent CUDA streams; fail closed if a CUDA experiment selects a non-CUDA/AOT backend.
2. Fresh baseline CPU4/GPU1, then CPU8/12/16 GPU1, separate runs under profiler. Select measured CPU configuration (at least8 to allow8 model calls) for GPU sweep2..8.
3. Every run uses same 64 regions, residual ROI and model settings; CPU/OpenCV threads unchanged. Compare full hashes, residual/protection/error semantics; 63 accepted plus existing overbroad page13 rejection expected. Each run new process and directory, external timeout300s.
4. Record wall/core/setup, RSS, approximate global VRAM and observed host-call overlap. Distinct streams enable scheduling; host-call overlap is not proof of simultaneous CUDA kernels. Do not claim batch inference or multiple GPUs.
5. Stop higher GPU counts after OOM/driver failure; do not silently substitute CPU/fallback. Repeat baseline/winner to assess variation. Save table and limits. No sum with preparation timings from another run.
