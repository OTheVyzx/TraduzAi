import { describe, expect, it } from "vitest";
import { readPsd } from "ag-psd";
import { writePsdRasterLayers } from "../psd";

function solid(width: number, height: number, rgba: [number, number, number, number]) {
  const pixels = new Uint8Array(width * height * 4);
  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = rgba[0];
    pixels[index + 1] = rgba[1];
    pixels[index + 2] = rgba[2];
    pixels[index + 3] = rgba[3];
  }
  return pixels;
}

function u16(bytes: Uint8Array, offset: number) {
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function i16(bytes: Uint8Array, offset: number) {
  const value = u16(bytes, offset);
  return value > 0x7fff ? value - 0x10000 : value;
}

function u32(bytes: Uint8Array, offset: number) {
  return ((bytes[offset] * 0x1000000) + (bytes[offset + 1] << 16) + (bytes[offset + 2] << 8) + bytes[offset + 3]) >>> 0;
}

function parseStructure(bytes: Uint8Array) {
  return readPsd(bytes, {
    skipLayerImageData: true,
    skipCompositeImageData: true,
    skipThumbnail: true,
  });
}

describe("Studio PSD export", () => {
  it("writes a PSD header and real layer section", () => {
    const bytes = writePsdRasterLayers(2, 2, [
      { name: "Original", pixels: solid(2, 2, [255, 255, 255, 255]) },
      { name: "Texto 1", pixels: solid(2, 2, [0, 0, 0, 0]) },
    ]);

    expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe("8BPS");
    expect(u16(bytes, 4)).toBe(1);
    expect(u16(bytes, 12)).toBe(3);
    expect(u32(bytes, 14)).toBe(2);
    expect(u32(bytes, 18)).toBe(2);

    const psd = parseStructure(bytes);
    expect(psd.children?.map((layer) => layer.name)).toEqual(["Texto 1", "Original"]);
  });

  it("writes TySh editable text metadata for text layers", () => {
    const bytes = writePsdRasterLayers(4, 4, [
      { name: "Original", pixels: solid(4, 4, [255, 255, 255, 255]) },
      {
        name: "Texto 1",
        pixels: solid(4, 4, [0, 0, 0, 0]),
        textSpec: {
          text: "TESTE QA",
          x: 1,
          y: 2,
          width: 3,
          height: 4,
          fontName: "ArialMT",
          fontSize: 18,
          color: [0, 0, 0, 255],
          vertical: false,
          justification: "center",
        },
      },
    ]);
    const ascii = new TextDecoder("latin1").decode(bytes);

    expect(ascii).toContain("8BIMTySh");
    expect(ascii).toContain("EngineData");
    expect(ascii).toContain("Txt ");
    expect(ascii).toContain("FontSet");
  });

  it("serializes the text font size in EngineData", () => {
    const bytes = writePsdRasterLayers(20, 20, [
      { name: "Original", pixels: solid(20, 20, [255, 255, 255, 255]) },
      {
        name: "Texto 1",
        pixels: solid(10, 8, [0, 0, 0, 0]),
        left: 2,
        top: 3,
        right: 12,
        bottom: 11,
        textSpec: {
          text: "TESTE QA",
          x: 2,
          y: 3,
          width: 10,
          height: 8,
          fontName: "Comic Neue",
          fontSize: 37,
          color: [15, 25, 35, 255],
          vertical: false,
          justification: "center",
        },
      },
    ]);
    const ascii = new TextDecoder("latin1").decode(bytes);

    expect(ascii).toContain("/FontSize 37");
    expect(ascii).not.toMatch(/\/FontSize\s+0(?:\D|$)/);
  });

  it("writes bounded text layer rectangles instead of 1x1 placeholders", () => {
    const bytes = writePsdRasterLayers(20, 20, [
      { name: "Original", pixels: solid(20, 20, [255, 255, 255, 255]) },
      {
        name: "Texto 1",
        pixels: solid(6, 4, [0, 0, 0, 0]),
        left: 5,
        top: 7,
        right: 11,
        bottom: 11,
        textSpec: {
          text: "TESTE QA",
          x: 5,
          y: 7,
          width: 6,
          height: 4,
          fontName: "ArialMT",
          fontSize: 18,
          color: [0, 0, 0, 255],
          vertical: false,
          justification: "center",
        },
      },
    ]);

    const psd = parseStructure(bytes);
    const textLayer = psd.children?.find((layer) => layer.name === "Texto 1");
    expect(textLayer).toMatchObject({ top: 7, left: 5, bottom: 11, right: 11 });
  });
});
