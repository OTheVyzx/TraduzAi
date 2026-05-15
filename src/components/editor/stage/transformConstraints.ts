export interface PageSize {
  width: number;
  height: number;
}

export interface TransformBox {
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number;
}

const MIN_TEXT_BOX_SIZE = 20;

function finiteOr(value: number, fallback: number) {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function clampTextTransformBox<T extends TransformBox>(
  box: T,
  pageSize: PageSize | null | undefined,
  minSize = MIN_TEXT_BOX_SIZE,
): T {
  const width = Math.max(minSize, finiteOr(box.width, minSize));
  const height = Math.max(minSize, finiteOr(box.height, minSize));
  let x = finiteOr(box.x, 0);
  let y = finiteOr(box.y, 0);

  if (pageSize && pageSize.width > 0 && pageSize.height > 0) {
    x = clamp(x, 0, Math.max(0, pageSize.width - width));
    y = clamp(y, 0, Math.max(0, pageSize.height - height));
  }

  return {
    ...box,
    x,
    y,
    width,
    height,
  };
}
