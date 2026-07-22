import { describe, expect, it } from "vitest";
import type { FluxGenerateConfig, FluxGenerateResult } from "../fluxContract";
import { createFluxStore } from "../fluxStore";

function config(jobId = "job-1"): FluxGenerateConfig {
  return {
    contract_version: "1.0",
    job_id: jobId,
    prompt: "reconstruir textura",
    negative_prompt: "texto",
    model: "flux-fill-local",
    source_png_data: "data:image/png;base64,source",
    mask_png_data: "data:image/png;base64,mask",
    width: 128,
    height: 96,
    variant_count: 2,
    seed: 42,
    steps: 20,
    guidance_scale: 18,
  };
}

function result(jobId = "job-1"): FluxGenerateResult {
  return {
    contract_version: "1.0",
    job_id: jobId,
    provider: "local-adapter",
    model: "flux-fill-local",
    variants: [
      { id: "a", seed: 42, png_data: "data:image/png;base64,a" },
      { id: "b", seed: 43, png_data: "data:image/png;base64,b" },
    ],
  };
}

describe("FLUX store", () => {
  it("tracks provider status and exposes local setup failures", async () => {
    const store = createFluxStore();
    await store.getState().checkProvider(async () => ({
      status: "missing",
      provider: "local-adapter",
      message: "Configure TRADUZAI_STUDIO_FLUX_COMMAND",
    }));

    expect(store.getState()).toMatchObject({
      isCheckingProvider: false,
      providerStatus: {
        status: "missing",
        provider: "local-adapter",
        message: "Configure TRADUZAI_STUDIO_FLUX_COMMAND",
      },
      error: null,
    });
  });

  it("keeps only a valid matching result for the active job", async () => {
    const store = createFluxStore();
    const generated = await store.getState().generate(config(), async () => result());

    expect(generated).toEqual(result());
    expect(store.getState().activeJob).toMatchObject({ status: "ready", result: result() });

    await expect(store.getState().generate(config("job-2"), async () => result("wrong-job")))
      .rejects.toThrow("job diferente");
    expect(store.getState().activeJob?.status).toBe("error");
  });

  it("ignores a late provider result after the user cancels the job", async () => {
    const store = createFluxStore();
    let resolve!: (value: FluxGenerateResult) => void;
    let finishCancellation!: () => void;
    const pending = new Promise<FluxGenerateResult>((next) => { resolve = next; });
    const cancellationGate = new Promise<void>((next) => { finishCancellation = next; });
    const generation = store.getState().generate(config(), () => pending);

    const cancellation = store.getState().cancelActiveJob(async (jobId) => {
      expect(jobId).toBe("job-1");
      await cancellationGate;
      return true;
    });
    expect(store.getState().activeJob?.status).toBe("cancelling");
    finishCancellation();
    await cancellation;
    resolve(result());

    await expect(generation).resolves.toBeNull();
    expect(store.getState().activeJob).toBeNull();
    expect(store.getState().error).toBeNull();
  });

  it("keeps the job blocked when the worker does not confirm cancellation", async () => {
    const store = createFluxStore();
    const pending = new Promise<FluxGenerateResult>(() => undefined);
    void store.getState().generate(config(), () => pending);

    await store.getState().cancelActiveJob(async () => false);
    expect(store.getState().activeJob).toMatchObject({
      status: "generating",
      error: "O worker FLUX não confirmou o cancelamento",
    });
    expect(store.getState().error).toContain("não confirmou");

    await store.getState().cancelActiveJob(async () => {
      throw new Error("timeout ao matar worker");
    });
    expect(store.getState().activeJob?.status).toBe("generating");
    expect(store.getState().error).toContain("timeout");
  });
});
