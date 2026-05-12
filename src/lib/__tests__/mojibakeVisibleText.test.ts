import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const ROOTS = ["src/pages", "src/components/editor"];
const FILE_RE = /\.(tsx?|jsx?)$/;
const MOJIBAKE_RE = /Ã|Â|â€”|â€|PÃ|CapÃ|traduÃ/;

function collectFiles(dir: string): string[] {
  const entries = readdirSync(dir);
  return entries.flatMap((entry) => {
    const path = join(dir, entry);
    const stat = statSync(path);
    if (stat.isDirectory()) return collectFiles(path);
    return FILE_RE.test(path) ? [path] : [];
  });
}

describe("textos visiveis sem mojibake", () => {
  it("nao deixa literais corrompidos em pages e editor", () => {
    const offenders = ROOTS.flatMap(collectFiles).filter((file) => {
      const text = readFileSync(file, "utf8");
      return MOJIBAKE_RE.test(text);
    });

    expect(offenders).toEqual([]);
  });
});
