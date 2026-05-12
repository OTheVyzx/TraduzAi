import { describe, expect, test } from "vitest";
import {
  EDITOR_TEXT_LINE_HEIGHT,
  fitEditorTextFontSize,
  wrapEditorText,
} from "../../components/editor/stage/textFit";

const measure = (text: string, _fontCss: string, fontSize: number) => text.length * fontSize * 0.52;

describe("fitEditorTextFontSize", () => {
  test("reduz o tamanho quando o texto traduzido nao cabe na caixa", () => {
    const text = "EVITAR ISSO SERIA A MELHOR ESCOLHA, SE POSSIVEL.";
    const fitted = fitEditorTextFontSize({
      text,
      fontFamily: "Comic Neue",
      fontStyle: "bold",
      maxFontSize: 47,
      maxWidth: 145,
      maxHeight: 72,
      measureText: measure,
    });

    expect(fitted).toBeLessThan(47);
    const lines = wrapEditorText(text, 145, `bold ${fitted}px "Comic Neue"`, fitted, measure);
    expect(lines.length * fitted * EDITOR_TEXT_LINE_HEIGHT).toBeLessThanOrEqual(72);
  });

  test("mantem o tamanho quando ja cabe", () => {
    expect(
      fitEditorTextFontSize({
        text: "TEXTO CURTO",
        fontFamily: "Comic Neue",
        fontStyle: "bold",
        maxFontSize: 28,
        maxWidth: 220,
        maxHeight: 80,
        measureText: measure,
      }),
    ).toBe(28);
  });
});
