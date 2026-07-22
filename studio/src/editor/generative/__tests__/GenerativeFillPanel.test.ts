import { describe, expect, it } from "vitest";
import { GENERATIVE_FILL_COPY } from "../GenerativeFillPanel";

describe("GenerativeFillPanel", () => {
  it("presents a Studio-only local FLUX workflow in Portuguese", () => {
    expect(GENERATIVE_FILL_COPY).toEqual({
      title: "Preenchimento FLUX",
      prompt: "Prompt opcional",
      defaultVariants: "2 variantes",
      localAdapter: "Configure o adaptador local para usar FLUX",
    });
    expect(Object.values(GENERATIVE_FILL_COPY).join(" ")).not.toMatch(/OCR|Traduzir/);
  });
});
