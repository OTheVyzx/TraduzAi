import { describe, expect, it } from "vitest";
import { clampTextTransformBox } from "../../../components/editor/stage/transformConstraints";

describe("editor transform constraints", () => {
  it("enforces minimum text box dimensions", () => {
    expect(
      clampTextTransformBox(
        { x: 10, y: 20, width: 4, height: 8 },
        { width: 500, height: 700 },
      ),
    ).toMatchObject({ x: 10, y: 20, width: 20, height: 20 });
  });

  it("keeps transformed text boxes inside the page bounds", () => {
    expect(
      clampTextTransformBox(
        { x: -12, y: 690, width: 80, height: 60 },
        { width: 500, height: 700 },
      ),
    ).toMatchObject({ x: 0, y: 640, width: 80, height: 60 });
  });
});
