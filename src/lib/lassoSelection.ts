export type LassoSelectionRegionOperation = "add" | "subtract";

export type LassoSelectionRegion = {
  operation: LassoSelectionRegionOperation;
  points: Array<[number, number]>;
};

export type LassoSelection = {
  pageKey: string;
  pageIndex: number;
  points: Array<[number, number]>;
  bbox: [number, number, number, number];
  width: number;
  height: number;
  id?: string;
  regions?: LassoSelectionRegion[];
  feather?: number;
  expansion?: number;
  targetNodeId?: string | null;
};

export const MAX_SELECTION_FEATHER = 128;
export const MAX_SELECTION_EXPANSION = 256;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function lassoBoundingBox(
  points: Array<[number, number]>,
  width?: number,
  height?: number,
): [number, number, number, number] {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const bbox: [number, number, number, number] = [
    Math.floor(Math.min(...xs)),
    Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)),
    Math.ceil(Math.max(...ys)),
  ];
  if (typeof width !== "number" || typeof height !== "number") return bbox;
  return [
    clamp(bbox[0], 0, width),
    clamp(bbox[1], 0, height),
    clamp(bbox[2], 0, width),
    clamp(bbox[3], 0, height),
  ];
}

export function createLassoSelection(input: {
  pageKey: string;
  pageIndex: number;
  points: Array<[number, number]>;
  width: number;
  height: number;
  id?: string;
  feather?: number;
  expansion?: number;
  targetNodeId?: string | null;
  regions?: LassoSelectionRegion[];
}): LassoSelection {
  const points = input.points.map(([x, y]) => [
    clamp(Math.round(x), 0, input.width),
    clamp(Math.round(y), 0, input.height),
  ] as [number, number]);
  const selection: LassoSelection = {
    pageKey: input.pageKey,
    pageIndex: input.pageIndex,
    points,
    bbox: lassoBoundingBox(points, input.width, input.height),
    width: input.width,
    height: input.height,
  };
  if (input.id !== undefined) selection.id = input.id;
  if (input.regions !== undefined) {
    selection.regions = input.regions.map((region) => ({
      operation: region.operation,
      points: region.points.map(([x, y]) => [
        clamp(Math.round(x), 0, input.width),
        clamp(Math.round(y), 0, input.height),
      ]),
    }));
  }
  if (input.feather !== undefined) {
    selection.feather = clamp(Math.round(input.feather), 0, MAX_SELECTION_FEATHER);
  }
  if (input.expansion !== undefined) {
    selection.expansion = clamp(Math.round(input.expansion), -MAX_SELECTION_EXPANSION, MAX_SELECTION_EXPANSION);
  }
  if (input.targetNodeId !== undefined) selection.targetNodeId = input.targetNodeId;
  return selection;
}

function selectionRegions(selection: LassoSelection): LassoSelectionRegion[] {
  if (selection.regions && selection.regions.length > 0) {
    return selection.regions.map((region) => ({
      operation: region.operation,
      points: region.points.map(([x, y]) => [x, y]),
    }));
  }
  return [{ operation: "add", points: selection.points.map(([x, y]) => [x, y]) }];
}

function bboxUnion(
  left: [number, number, number, number],
  right: [number, number, number, number],
): [number, number, number, number] {
  return [
    Math.min(left[0], right[0]),
    Math.min(left[1], right[1]),
    Math.max(left[2], right[2]),
    Math.max(left[3], right[3]),
  ];
}

export function combineLassoSelections(
  current: LassoSelection | null,
  next: LassoSelection,
  operation: "replace" | LassoSelectionRegionOperation,
): LassoSelection {
  if (
    !current
    || operation === "replace"
    || current.pageKey !== next.pageKey
    || current.pageIndex !== next.pageIndex
    || current.width !== next.width
    || current.height !== next.height
  ) {
    return next;
  }

  const nextRegion: LassoSelectionRegion = {
    operation,
    points: next.points.map(([x, y]) => [x, y]),
  };
  return {
    ...current,
    points: next.points.map(([x, y]) => [x, y]),
    bbox: operation === "add" ? bboxUnion(current.bbox, next.bbox) : current.bbox,
    regions: [...selectionRegions(current), nextRegion],
    targetNodeId: next.targetNodeId ?? current.targetNodeId ?? null,
  };
}

export function withLassoSelectionModifiers(
  selection: LassoSelection,
  patch: {
    feather?: number;
    expansion?: number;
    targetNodeId?: string | null;
  },
): LassoSelection {
  return {
    ...selection,
    ...(patch.feather !== undefined
      ? { feather: clamp(Math.round(patch.feather), 0, MAX_SELECTION_FEATHER) }
      : {}),
    ...(patch.expansion !== undefined
      ? { expansion: clamp(Math.round(patch.expansion), -MAX_SELECTION_EXPANSION, MAX_SELECTION_EXPANSION) }
      : {}),
    ...(patch.targetNodeId !== undefined ? { targetNodeId: patch.targetNodeId } : {}),
  };
}

export function lassoSelectionEffectiveBbox(
  selection: LassoSelection,
): [number, number, number, number] {
  const expansion = selection.expansion ?? 0;
  const feather = selection.feather ?? 0;
  const delta = expansion + feather;
  const [x1, y1, x2, y2] = selection.bbox;
  const adjusted: [number, number, number, number] = [
    clamp(Math.floor(x1 - delta), 0, selection.width),
    clamp(Math.floor(y1 - delta), 0, selection.height),
    clamp(Math.ceil(x2 + delta), 0, selection.width),
    clamp(Math.ceil(y2 + delta), 0, selection.height),
  ];
  if (adjusted[2] < adjusted[0]) {
    const center = clamp(Math.round((x1 + x2) / 2), 0, selection.width);
    adjusted[0] = center;
    adjusted[2] = center;
  }
  if (adjusted[3] < adjusted[1]) {
    const center = clamp(Math.round((y1 + y2) / 2), 0, selection.height);
    adjusted[1] = center;
    adjusted[3] = center;
  }
  return adjusted;
}

export function lassoSelectionProcessingBbox(
  selection: LassoSelection,
): [number, number, number, number] {
  const margin = Math.max(0, selection.expansion ?? 0) + (selection.feather ?? 0);
  return [
    clamp(Math.floor(selection.bbox[0] - margin), 0, selection.width),
    clamp(Math.floor(selection.bbox[1] - margin), 0, selection.height),
    clamp(Math.ceil(selection.bbox[2] + margin), 0, selection.width),
    clamp(Math.ceil(selection.bbox[3] + margin), 0, selection.height),
  ];
}

function extremaLine(values: Uint8ClampedArray, radius: number, useMaximum: boolean) {
  if (radius <= 0) return new Uint8ClampedArray(values);
  const output = new Uint8ClampedArray(values.length);
  const windowSize = radius * 2 + 1;
  const dequeIndexes = new Int32Array(values.length + radius * 2);
  const dequeValues = new Uint8ClampedArray(values.length + radius * 2);
  let head = 0;
  let tail = 0;

  for (let paddedIndex = 0; paddedIndex < values.length + radius * 2; paddedIndex += 1) {
    const sourceIndex = paddedIndex - radius;
    const value = sourceIndex >= 0 && sourceIndex < values.length ? values[sourceIndex] : 0;
    while (head < tail && dequeIndexes[head] <= paddedIndex - windowSize) head += 1;
    while (
      head < tail
      && (useMaximum ? dequeValues[tail - 1] <= value : dequeValues[tail - 1] >= value)
    ) {
      tail -= 1;
    }
    dequeIndexes[tail] = paddedIndex;
    dequeValues[tail] = value;
    tail += 1;
    if (paddedIndex >= windowSize - 1) {
      const outputIndex = paddedIndex - (windowSize - 1);
      if (outputIndex < output.length) output[outputIndex] = dequeValues[head];
    }
  }
  return output;
}

function morphology(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  radius: number,
  useMaximum: boolean,
) {
  if (radius <= 0) return new Uint8ClampedArray(source);
  const horizontal = new Uint8ClampedArray(source.length);
  for (let y = 0; y < height; y += 1) {
    const row = source.slice(y * width, (y + 1) * width);
    horizontal.set(extremaLine(row, radius, useMaximum), y * width);
  }
  const output = new Uint8ClampedArray(source.length);
  for (let x = 0; x < width; x += 1) {
    const column = new Uint8ClampedArray(height);
    for (let y = 0; y < height; y += 1) column[y] = horizontal[y * width + x];
    const filtered = extremaLine(column, radius, useMaximum);
    for (let y = 0; y < height; y += 1) output[y * width + x] = filtered[y];
  }
  return output;
}

function boxBlurLine(values: Uint8ClampedArray, radius: number) {
  if (radius <= 0) return new Uint8ClampedArray(values);
  const output = new Uint8ClampedArray(values.length);
  const windowSize = radius * 2 + 1;
  let sum = 0;
  for (let index = -radius; index <= radius; index += 1) {
    if (index >= 0 && index < values.length) sum += values[index];
  }
  for (let index = 0; index < values.length; index += 1) {
    output[index] = Math.round(sum / windowSize);
    const leaving = index - radius;
    const entering = index + radius + 1;
    if (leaving >= 0 && leaving < values.length) sum -= values[leaving];
    if (entering >= 0 && entering < values.length) sum += values[entering];
  }
  return output;
}

function boxBlur(source: Uint8ClampedArray, width: number, height: number, radius: number) {
  if (radius <= 0) return new Uint8ClampedArray(source);
  const horizontal = new Uint8ClampedArray(source.length);
  for (let y = 0; y < height; y += 1) {
    horizontal.set(boxBlurLine(source.slice(y * width, (y + 1) * width), radius), y * width);
  }
  const output = new Uint8ClampedArray(source.length);
  for (let x = 0; x < width; x += 1) {
    const column = new Uint8ClampedArray(height);
    for (let y = 0; y < height; y += 1) column[y] = horizontal[y * width + x];
    const blurred = boxBlurLine(column, radius);
    for (let y = 0; y < height; y += 1) output[y * width + x] = blurred[y];
  }
  return output;
}

export function applySelectionAlphaModifiers(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  modifiers: { expansion?: number; feather?: number },
) {
  if (source.length !== width * height) {
    throw new Error("A máscara alfa não corresponde às dimensões da seleção");
  }
  const expansion = clamp(
    Math.round(modifiers.expansion ?? 0),
    -MAX_SELECTION_EXPANSION,
    MAX_SELECTION_EXPANSION,
  );
  const feather = clamp(Math.round(modifiers.feather ?? 0), 0, MAX_SELECTION_FEATHER);
  let output = new Uint8ClampedArray(source);
  if (expansion !== 0) {
    output = morphology(output, width, height, Math.abs(expansion), expansion > 0);
  }
  if (feather > 0) output = boxBlur(output, width, height, feather);
  return output;
}

function drawSelectionRegions(ctx: CanvasRenderingContext2D, selection: LassoSelection) {
  for (const region of selectionRegions(selection)) {
    if (region.points.length < 3) continue;
    ctx.globalCompositeOperation = region.operation === "subtract" ? "destination-out" : "source-over";
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(region.points[0][0], region.points[0][1]);
    for (let index = 1; index < region.points.length; index += 1) {
      ctx.lineTo(region.points[index][0], region.points[index][1]);
    }
    ctx.closePath();
    ctx.fill();
  }
}

export function rasterizeLassoSelectionToCanvas(selection: LassoSelection): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = selection.width;
  canvas.height = selection.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || selection.points.length < 3) return null;
  drawSelectionRegions(ctx, selection);

  if ((selection.expansion ?? 0) !== 0 || (selection.feather ?? 0) !== 0) {
    const [x1, y1, x2, y2] = lassoSelectionProcessingBbox(selection);
    const processingWidth = Math.max(1, x2 - x1);
    const processingHeight = Math.max(1, y2 - y1);
    const image = ctx.getImageData(x1, y1, processingWidth, processingHeight);
    const alpha = new Uint8ClampedArray(processingWidth * processingHeight);
    for (let index = 0; index < alpha.length; index += 1) alpha[index] = image.data[index * 4 + 3];
    const transformed = applySelectionAlphaModifiers(alpha, processingWidth, processingHeight, selection);
    for (let index = 0; index < transformed.length; index += 1) {
      image.data[index * 4] = 255;
      image.data[index * 4 + 1] = 255;
      image.data[index * 4 + 2] = 255;
      image.data[index * 4 + 3] = transformed[index];
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.putImageData(image, x1, y1);
  }
  return canvas;
}

export function rasterizeLassoSelectionToPng(selection: LassoSelection): string {
  return rasterizeLassoSelectionToCanvas(selection)?.toDataURL("image/png") ?? "";
}

export function rasterizeLassoToPng(points: Array<[number, number]>, width: number, height: number): string {
  return rasterizeLassoSelectionToPng({
    pageKey: "",
    pageIndex: 0,
    points,
    bbox: lassoBoundingBox(points, width, height),
    width,
    height,
  });
}
