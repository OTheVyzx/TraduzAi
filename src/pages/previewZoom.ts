export interface PreviewPanOffset {
  x: number;
  y: number;
}

export interface PreviewPanSession {
  startX: number;
  startY: number;
  originX: number;
  originY: number;
}

export const PREVIEW_ZOOM_DEFAULT = 1;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 6;
const ZOOM_STEP = 0.15;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function getNextPreviewZoom(currentZoom: number, direction: "in" | "out"): number {
  const delta = direction === "in" ? ZOOM_STEP : -ZOOM_STEP;
  return Number(clamp(currentZoom + delta, MIN_ZOOM, MAX_ZOOM).toFixed(2));
}

export function getDraggedPreviewPan(
  session: PreviewPanSession,
  event: MouseEvent,
): PreviewPanOffset {
  return {
    x: session.originX + (event.clientX - session.startX),
    y: session.originY + (event.clientY - session.startY),
  };
}

export function getPreviewWheelState(params: {
  zoom: number;
  pan: PreviewPanOffset;
  deltaX: number;
  deltaY: number;
  withZoomModifier: boolean;
}): { zoom: number; pan: PreviewPanOffset } {
  if (params.withZoomModifier) {
    const direction = params.deltaY < 0 ? "in" : "out";
    return {
      zoom: getNextPreviewZoom(params.zoom, direction),
      pan: params.pan,
    };
  }

  return {
    zoom: params.zoom,
    pan: {
      x: params.pan.x - params.deltaX,
      y: params.pan.y - params.deltaY,
    },
  };
}
