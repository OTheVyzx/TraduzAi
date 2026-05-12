export const EDITOR_TEXT_LINE_HEIGHT = 1.12;

type MeasureText = (text: string, fontCss: string, fontSize: number) => number;

type FitTextFontSizeOptions = {
  text: string;
  fontFamily: string;
  fontStyle: string;
  maxFontSize: number;
  maxWidth: number;
  maxHeight: number;
  minFontSize?: number;
  lineHeight?: number;
  measureText?: MeasureText;
};

let sharedContext: CanvasRenderingContext2D | null = null;

function defaultMeasureText(text: string, fontCss: string): number {
  if (typeof document === "undefined") return text.length * 12;
  if (!sharedContext) {
    sharedContext = document.createElement("canvas").getContext("2d");
  }
  if (!sharedContext) return text.length * 12;
  sharedContext.font = fontCss;
  return sharedContext.measureText(text).width;
}

function fontCss(fontStyle: string, fontSize: number, fontFamily: string) {
  return `${fontStyle || "normal"} ${fontSize}px "${fontFamily}"`;
}

function splitLongWord(
  word: string,
  maxWidth: number,
  font: string,
  fontSize: number,
  measureText: MeasureText,
): string[] {
  const parts: string[] = [];
  let current = "";

  for (const char of Array.from(word)) {
    const candidate = `${current}${char}`;
    if (current && measureText(candidate, font, fontSize) > maxWidth) {
      parts.push(current);
      current = char;
    } else {
      current = candidate;
    }
  }

  if (current) parts.push(current);
  return parts;
}

export function wrapEditorText(
  text: string,
  maxWidth: number,
  font: string,
  fontSize: number,
  measureText: MeasureText = defaultMeasureText,
): string[] {
  if (!text.trim()) return [];

  const lines: string[] = [];
  const paragraphs = text.replace(/\r\n/g, "\n").split("\n");

  for (const paragraph of paragraphs) {
    const words = paragraph.trim().split(/\s+/).filter(Boolean);
    if (words.length === 0) {
      lines.push("");
      continue;
    }

    let line = "";
    for (const rawWord of words) {
      const wordParts =
        measureText(rawWord, font, fontSize) > maxWidth
          ? splitLongWord(rawWord, maxWidth, font, fontSize, measureText)
          : [rawWord];

      for (const word of wordParts) {
        const candidate = line ? `${line} ${word}` : word;
        if (line && measureText(candidate, font, fontSize) > maxWidth) {
          lines.push(line);
          line = word;
        } else {
          line = candidate;
        }
      }
    }

    if (line) lines.push(line);
  }

  return lines;
}

function fitsAtSize({
  text,
  fontFamily,
  fontStyle,
  maxWidth,
  maxHeight,
  lineHeight,
  fontSize,
  measureText,
}: Omit<Required<FitTextFontSizeOptions>, "maxFontSize" | "minFontSize"> & { fontSize: number }) {
  const font = fontCss(fontStyle, fontSize, fontFamily);
  const lines = wrapEditorText(text, maxWidth, font, fontSize, measureText);
  if (lines.length === 0) return true;

  const widest = Math.max(...lines.map((line) => measureText(line, font, fontSize)));
  const totalHeight = lines.length * fontSize * lineHeight;
  return widest <= maxWidth && totalHeight <= maxHeight;
}

export function fitEditorTextFontSize({
  text,
  fontFamily,
  fontStyle,
  maxFontSize,
  maxWidth,
  maxHeight,
  minFontSize = 8,
  lineHeight = EDITOR_TEXT_LINE_HEIGHT,
  measureText = defaultMeasureText,
}: FitTextFontSizeOptions): number {
  const hi = Math.max(minFontSize, Math.floor(maxFontSize));
  const width = Math.max(1, Math.floor(maxWidth));
  const height = Math.max(1, Math.floor(maxHeight));

  if (!text.trim() || width <= 1 || height <= 1) return hi;
  if (fitsAtSize({ text, fontFamily, fontStyle, maxWidth: width, maxHeight: height, lineHeight, fontSize: hi, measureText })) {
    return hi;
  }

  let lo = minFontSize;
  let best = minFontSize;
  let right = hi - 1;

  while (lo <= right) {
    const mid = Math.floor((lo + right) / 2);
    if (fitsAtSize({ text, fontFamily, fontStyle, maxWidth: width, maxHeight: height, lineHeight, fontSize: mid, measureText })) {
      best = mid;
      lo = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return best;
}
