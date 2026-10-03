import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const editorSource = readFileSync(new URL("../../pages/Editor.tsx", import.meta.url), "utf8");

describe("editor header navigation", () => {
  it("uses the left header arrow to return directly to the first page", () => {
    expect(editorSource).toContain("onClick={() => triggerPageChange(0)}");
    expect(editorSource).toContain('title="Ir para a primeira página"');
  });
});
