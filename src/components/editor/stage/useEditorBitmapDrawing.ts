import { useCallback, useEffect, useRef } from "react";
import type { Bbox } from "../../../lib/editorHistory";

type BitmapLayerKey = "brush" | "mask";

type BitmapStrokePayload = {
  pageKey?: string;
  pageIndex?: number;
  width: number;
  height: number;
  strokes: [number, number][][];
  clear?: boolean;
  layerKey?: "brush" | "mask" | "recovery" | "reinpaint";
  erase?: boolean;
  brushSize?: number;
  color?: string;
  opacity?: number;
  hardness?: number;
  optimisticPath?: string;
  pngData?: string;
  clipMaskPng?: string;
  dirty_bbox?: Bbox;
};

type BitmapDrawingDeps = {
  pageKey: string;
  pageIndex: number;
  width: number;
  height: number;
  applyBitmapStroke: (payload: BitmapStrokePayload) => Promise<void>;
  healPaintedRegion: (payload: { pageKey?: string; pageIndex?: number; bbox: Bbox; maskPath?: string; maskPngData?: string }) => Promise<void>;
};

type QueuedPageContext = {
  pageKey: string;
  pageIndex: number;
  width: number;
  height: number;
};

type QueuedRun = (context: QueuedPageContext) => Promise<void>;

export function useEditorBitmapDrawing(deps: BitmapDrawingDeps) {
  const depsRef = useRef(deps);
  const bitmapPersistQueueRef = useRef<Partial<Record<BitmapLayerKey, Promise<void>>>>({});
  const recoveryPersistQueueRef = useRef<Promise<void>>(Promise.resolve());
  depsRef.current = deps;

  useEffect(() => {
    bitmapPersistQueueRef.current = {};
    recoveryPersistQueueRef.current = Promise.resolve();
  }, [deps.pageKey, deps.pageIndex, deps.width, deps.height]);

  const capturePageContext = useCallback(
    (): QueuedPageContext => ({
      pageKey: depsRef.current.pageKey,
      pageIndex: depsRef.current.pageIndex,
      width: depsRef.current.width,
      height: depsRef.current.height,
    }),
    [],
  );

  const runQueued = useCallback(async (context: QueuedPageContext, run: QueuedRun) => {
    await run(context);
  }, []);

  const enqueueBitmapPersist = useCallback((layerKey: BitmapLayerKey, run: QueuedRun) => {
    const context = capturePageContext();
    const queued = (bitmapPersistQueueRef.current[layerKey] ?? Promise.resolve())
      .catch(() => undefined)
      .then(() => runQueued(context, run));
    bitmapPersistQueueRef.current[layerKey] = queued;
    return queued;
  }, [capturePageContext, runQueued]);

  const enqueueRecoveryPersist = useCallback((run: QueuedRun) => {
    const context = capturePageContext();
    const queued = recoveryPersistQueueRef.current.catch(() => undefined).then(() => runQueued(context, run));
    recoveryPersistQueueRef.current = queued;
    return queued;
  }, [capturePageContext, runQueued]);

  return {
    enqueueBitmapPersist,
    enqueueRecoveryPersist,
    applyBitmapStroke: deps.applyBitmapStroke,
    healPaintedRegion: deps.healPaintedRegion,
  };
}
