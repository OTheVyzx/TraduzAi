export type LassoSelection = {
  pageKey: string;
  pageIndex: number;
  points: Array<[number, number]>;
  bbox: [number, number, number, number];
  width: number;
  height: number;
};

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
}): LassoSelection {
  const points = input.points.map(([x, y]) => [
    clamp(Math.round(x), 0, input.width),
    clamp(Math.round(y), 0, input.height),
  ] as [number, number]);
  return {
    pageKey: input.pageKey,
    pageIndex: input.pageIndex,
    points,
    bbox: lassoBoundingBox(points, input.width, input.height),
    width: input.width,
    height: input.height,
  };
}

export function rasterizeLassoToPng(points: Array<[number, number]>, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx || points.length < 3) return "";

  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();
  ctx.fill();
  return canvas.toDataURL("image/png");
}
