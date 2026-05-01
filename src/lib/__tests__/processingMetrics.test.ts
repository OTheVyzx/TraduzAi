import { describe, expect, it } from "vitest";
import { countFlagLogs, hardwareUsageLabel, pagesPerMinute, PERCEIVED_PROCESSING_STEPS } from "../processingMetrics";

describe("processingMetrics", () => {
  it("keeps the perceived processing steps explicit", () => {
    expect(PERCEIVED_PROCESSING_STEPS).toContain("Aplicando glossario");
    expect(PERCEIVED_PROCESSING_STEPS).toContain("Rodando QA");
  });

  it("calculates pages per minute and flag logs", () => {
    expect(pagesPerMinute({ current_page: 3 } as any, 90)).toBe(2);
    expect(countFlagLogs([{ message: "QA encontrou 2 flags" } as any, { message: "ok" } as any])).toBe(1);
  });

  it("formats hardware usage", () => {
    expect(hardwareUsageLabel(null)).toBe("Hardware em deteccao");
    expect(hardwareUsageLabel({ gpu_available: true, gpu_name: "RTX" } as any)).toBe("GPU ativa: RTX");
  });
});
