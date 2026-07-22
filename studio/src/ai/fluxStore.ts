import { create } from "zustand";
import type {
  FluxGenerateConfig,
  FluxGenerateResult,
  FluxProviderStatus,
} from "./fluxContract";

export interface FluxActiveJob {
  runId: number;
  status: "generating" | "cancelling" | "ready" | "error";
  request: FluxGenerateConfig;
  result: FluxGenerateResult | null;
  error: string | null;
}

export interface FluxState {
  providerStatus: FluxProviderStatus | null;
  isCheckingProvider: boolean;
  activeJob: FluxActiveJob | null;
  error: string | null;
  checkProvider: (loader: () => Promise<FluxProviderStatus>) => Promise<FluxProviderStatus | null>;
  generate: (
    request: FluxGenerateConfig,
    runner: (request: FluxGenerateConfig) => Promise<FluxGenerateResult>,
  ) => Promise<FluxGenerateResult | null>;
  cancelActiveJob: (canceller: (jobId: string) => Promise<boolean>) => Promise<void>;
  clearCompletedJob: () => void;
  clearError: () => void;
}

let nextRunId = 1;

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function validateFluxResult(request: FluxGenerateConfig, result: FluxGenerateResult) {
  if (result.contract_version !== "1.0") throw new Error("Versão de contrato FLUX incompatível");
  if (result.job_id !== request.job_id) throw new Error("O adaptador FLUX respondeu para um job diferente");
  if (result.variants.length < 2 || result.variants.length > 4) {
    throw new Error("O adaptador FLUX precisa retornar entre 2 e 4 variantes");
  }
  if (result.variants.length !== request.variant_count) {
    throw new Error("O adaptador FLUX retornou uma quantidade inesperada de variantes");
  }
  const ids = new Set<string>();
  for (const variant of result.variants) {
    if (!variant.id.trim() || ids.has(variant.id)) throw new Error("O adaptador FLUX retornou variantes sem ids únicos");
    if (!variant.png_data?.trim() && !variant.path?.trim()) {
      throw new Error(`A variante FLUX ${variant.id} não possui imagem`);
    }
    ids.add(variant.id);
  }
  return result;
}

export function createFluxStore() {
  return create<FluxState>((set, get) => ({
    providerStatus: null,
    isCheckingProvider: false,
    activeJob: null,
    error: null,

    checkProvider: async (loader) => {
      set({ isCheckingProvider: true, error: null });
      try {
        const providerStatus = await loader();
        set({ providerStatus, isCheckingProvider: false, error: null });
        return providerStatus;
      } catch (error) {
        const message = errorMessage(error);
        set({
          providerStatus: { status: "error", provider: "local-adapter", message },
          isCheckingProvider: false,
          error: message,
        });
        return null;
      }
    },

    generate: async (request, runner) => {
      if (get().activeJob?.status === "generating") {
        throw new Error("Já existe uma geração FLUX em andamento");
      }
      const runId = nextRunId++;
      set({
        activeJob: { runId, status: "generating", request, result: null, error: null },
        error: null,
      });
      try {
        const result = validateFluxResult(request, await runner(request));
        if (get().activeJob?.runId !== runId || get().activeJob?.status === "cancelling") return null;
        set({
          activeJob: { runId, status: "ready", request, result, error: null },
          error: null,
        });
        return result;
      } catch (error) {
        if (get().activeJob?.runId !== runId || get().activeJob?.status === "cancelling") return null;
        const message = errorMessage(error);
        set({
          activeJob: { runId, status: "error", request, result: null, error: message },
          error: message,
        });
        throw error;
      }
    },

    cancelActiveJob: async (canceller) => {
      const activeJob = get().activeJob;
      if (!activeJob) return;
      try {
        if (activeJob.status === "generating") {
          set({ activeJob: { ...activeJob, status: "cancelling" }, error: null });
          const confirmed = await canceller(activeJob.request.job_id);
          if (!confirmed) throw new Error("O worker FLUX não confirmou o cancelamento");
        }
        if (get().activeJob?.runId === activeJob.runId) set({ activeJob: null, error: null });
      } catch (error) {
        const message = errorMessage(error);
        if (get().activeJob?.runId === activeJob.runId) {
          set({
            activeJob: { ...activeJob, status: "generating", error: message },
            error: message,
          });
        }
      }
    },
    clearCompletedJob: () => {
      if (get().activeJob?.status !== "generating") set({ activeJob: null, error: null });
    },
    clearError: () => set({ error: null }),
  }));
}

export const useFluxStore = createFluxStore();
