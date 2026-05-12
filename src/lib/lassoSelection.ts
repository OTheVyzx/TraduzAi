export type LassoSelection = {
  pageKey: string;
  pageIndex: number;
  points: Array<[number, number]>;
  bbox: [number, number, number, number];
  width: number;
  height: number;
};

export function lassoBoundingBox(points: Array<[number, number]>): [number, number, number, number] {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return [
    Math.floor(Math.min(...xs)),
    Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)),
    Math.ceil(Math.max(...ys)),
  ];
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
