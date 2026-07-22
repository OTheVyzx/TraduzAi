export type EditorSceneVisualNode =
  | {
      id: string;
      kind: "bitmap";
      source: string;
      opacity: number;
      blendMode: string;
    }
  | {
      id: string;
      kind: "text";
      textLayerId: string;
    };
