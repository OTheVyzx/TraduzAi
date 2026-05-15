import { describe, expect, it } from "vitest";
import { formatPipelineQualityLabel, normalizePipelineQuality } from "../pipelineQuality";

describe("pipelineQuality", () => {
  it("normalizes legacy fast/default values to normal", () => {
    expect(normalizePipelineQuality("rapida")).toBe("normal");
    expect(normalizePipelineQuality("normal")).toBe("normal");
    expect(normalizePipelineQuality(undefined)).toBe("normal");
  });

  it("normalizes legacy high values to ultra", () => {
    expect(normalizePipelineQuality("alta")).toBe("ultra");
    expect(normalizePipelineQuality("max")).toBe("ultra");
    expect(normalizePipelineQuality("Ultra")).toBe("ultra");
  });

  it("formats only user-facing Normal and Ultra labels", () => {
    expect(formatPipelineQualityLabel("rapida")).toBe("Normal");
    expect(formatPipelineQualityLabel("alta")).toBe("Ultra");
  });
});
