import { useEffect, useRef, useState } from "react";
import Konva from "konva";
import { Group, Rect } from "react-konva";
import type { TextEntry } from "../../../lib/stores/appStore";
import type { TextTransformSnapshot } from "../../../lib/stores/editorStore";
import { bboxToRect } from "./coordinateUtils";
import { styleForLayer } from "./textLayerStyleUtils";

const ROTATE_HIT_MARGIN = 22;
const ROTATE_ANCHOR_CLEARANCE = 4;
const ROTATE_CURSOR =
  'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'24\' height=\'24\' viewBox=\'0 0 24 24\'%3E%3Cpath d=\'M20 11a8 8 0 1 0-2.34 5.66\' fill=\'none\' stroke=\'white\' stroke-width=\'5\' stroke-linecap=\'round\'/%3E%3Cpath d=\'M20 4v7h-7\' fill=\'none\' stroke=\'white\' stroke-width=\'5\' stroke-linecap=\'round\' stroke-linejoin=\'round\'/%3E%3Cpath d=\'M20 11a8 8 0 1 0-2.34 5.66\' fill=\'none\' stroke=\'black\' stroke-width=\'2\' stroke-linecap=\'round\'/%3E%3Cpath d=\'M20 4v7h-7\' fill=\'none\' stroke=\'black\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'/%3E%3C/svg%3E") 12 12, grab';

type RotateSession = {
  stage: Konva.Stage;
  center: { x: number; y: number };
  startAngle: number;
  startRotation: number;
};

export function EditorRotationHotspots({
  entry,
  draftRotation,
  onDraftRotation,
  onCommitTransform,
}: {
  entry: TextEntry;
  draftRotation: number | null;
  onDraftRotation: (rotation: number | null) => void;
  onCommitTransform: (before: TextTransformSnapshot, after: TextTransformSnapshot) => void;
}) {
  const [rotateSession, setRotateSession] = useState<RotateSession | null>(null);
  const finalRotationRef = useRef<number | null>(null);
  const latestRef = useRef({ bbox: entry.layout_bbox ?? entry.bbox, rotation: 0, onDraftRotation, onCommitTransform });

  const bbox = entry.layout_bbox ?? entry.bbox;
  const rect = bboxToRect(bbox);
  const rotation = normalizeRotationDegrees(styleForLayer(entry).rotacao);
  const displayedRotation = normalizeRotationDegrees(draftRotation ?? rotation);
  const zoneHeight = Math.max(1, rect.height - ROTATE_HIT_MARGIN * 2);
  const zoneWidth = ROTATE_HIT_MARGIN - ROTATE_ANCHOR_CLEARANCE;

  latestRef.current = { bbox, rotation, onDraftRotation, onCommitTransform };

  useEffect(() => {
    if (!rotateSession) return;

    const stageContainer = rotateSession.stage.container();
    stageContainer.style.cursor = ROTATE_CURSOR;
    document.body.style.cursor = ROTATE_CURSOR;

    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      const point = pointFromClient(rotateSession.stage, event);
      if (!point) return;
      const nextRotation = normalizeRotationDegrees(
        rotateSession.startRotation + angleDeltaDegrees(rotateSession.startAngle, angleFromCenter(rotateSession.center, point)),
      );
      finalRotationRef.current = nextRotation;
      latestRef.current.onDraftRotation(nextRotation);
    };

    const handleMouseUp = () => {
      const nextRotation = normalizeRotationDegrees(finalRotationRef.current ?? rotateSession.startRotation);
      const { bbox: latestBbox, onCommitTransform: commitTransform } = latestRef.current;
      if (Math.abs(rotateSession.startRotation - nextRotation) >= 0.01) {
        commitTransform(
          { bbox: latestBbox, rotacao: rotateSession.startRotation },
          { bbox: latestBbox, rotacao: nextRotation },
        );
      }
      finalRotationRef.current = null;
      latestRef.current.onDraftRotation(null);
      setRotateSession(null);
      stageContainer.style.cursor = "";
      document.body.style.cursor = "";
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp, { once: true });
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      stageContainer.style.cursor = "";
      document.body.style.cursor = "";
    };
  }, [rotateSession]);

  if (entry.visible === false || entry.locked) return null;

  const beginRotate = (stage: Konva.Stage | null, event: MouseEvent) => {
    const point = pointFromClient(stage, event);
    if (!stage || !point) return;
    const center = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
    setRotateSession({
      stage,
      center,
      startAngle: angleFromCenter(center, point),
      startRotation: displayedRotation,
    });
  };

  return (
    <Group
      x={rect.x + rect.width / 2}
      y={rect.y + rect.height / 2}
      offsetX={rect.width / 2}
      offsetY={rect.height / 2}
      width={rect.width}
      height={rect.height}
      rotation={displayedRotation}
      listening
    >
      <Rect
        x={-ROTATE_HIT_MARGIN}
        y={ROTATE_HIT_MARGIN}
        width={zoneWidth}
        height={zoneHeight}
        fill="rgba(255,255,255,0.001)"
        strokeEnabled={false}
        onMouseEnter={(event) => setStageCursor(event.target.getStage(), ROTATE_CURSOR)}
        onMouseLeave={(event) => {
          if (!rotateSession) setStageCursor(event.target.getStage(), "");
        }}
        onMouseDown={(event) => {
          event.cancelBubble = true;
          beginRotate(event.target.getStage(), event.evt);
        }}
      />
      <Rect
        x={rect.width + ROTATE_ANCHOR_CLEARANCE}
        y={ROTATE_HIT_MARGIN}
        width={zoneWidth}
        height={zoneHeight}
        fill="rgba(255,255,255,0.001)"
        strokeEnabled={false}
        onMouseEnter={(event) => setStageCursor(event.target.getStage(), ROTATE_CURSOR)}
        onMouseLeave={(event) => {
          if (!rotateSession) setStageCursor(event.target.getStage(), "");
        }}
        onMouseDown={(event) => {
          event.cancelBubble = true;
          beginRotate(event.target.getStage(), event.evt);
        }}
      />
    </Group>
  );
}

function setStageCursor(stage: Konva.Stage | null, cursor: string) {
  if (!stage) return;
  stage.container().style.cursor = cursor;
}

function pointFromClient(stage: Konva.Stage | null, event: MouseEvent) {
  if (!stage) return null;
  const rect = stage.container().getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  return {
    x: ((event.clientX - rect.left) / rect.width) * stage.width(),
    y: ((event.clientY - rect.top) / rect.height) * stage.height(),
  };
}

function angleFromCenter(center: { x: number; y: number }, point: { x: number; y: number }) {
  return (Math.atan2(point.y - center.y, point.x - center.x) * 180) / Math.PI;
}

function angleDeltaDegrees(start: number, current: number) {
  let delta = current - start;
  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;
  return delta;
}

function normalizeRotationDegrees(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  let normalized = numeric % 360;
  if (normalized > 180) normalized -= 360;
  if (normalized <= -180) normalized += 360;
  if (Math.abs(normalized) < 0.01) return 0;
  return Math.round(normalized * 100) / 100;
}
