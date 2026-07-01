import { describe, expect, it } from "vitest";
import type { StudioPage } from "../../project/studioProject";
import { bboxToPercentStyle, inferPageSize, readableLayerLabel } from "../pageGeometry";

describe("Studio editor geometry", () => {
  it("infers a stable page size from text boxes", () => {
    const page = {
      text_layers: [
        { bbox: [10, 20, 1200, 1600] },
      ],
    } as StudioPage;

    expect(inferPageSize(page)).toEqual({ width: 1200, height: 1600 });
    expect(inferPageSize(null)).toEqual({ width: 900, height: 1280 });
  });

  it("converts bbox coordinates to percent styles", () => {
    expect(bboxToPercentStyle([90, 128, 180, 256], { width: 900, height: 1280 })).toEqual({
      left: "10%",
      top: "10%",
      width: "10%",
      height: "10%",
    });
  });

  it("formats layer labels", () => {
    expect(readableLayerLabel(0)).toBe("Texto 01");
    expect(readableLayerLabel(11)).toBe("Texto 12");
  });
});
