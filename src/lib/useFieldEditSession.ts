import { useCallback, useRef } from "react";
import { useEditorStore } from "./stores/editorStore";
import type { TextLayerStyle } from "./stores/appStore";

function commandId(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`;
}

type EditorStoreState = ReturnType<typeof useEditorStore.getState>;

export function selectFieldEditValue<K extends keyof TextLayerStyle>(
  state: EditorStoreState,
  pageKey: string,
  layerId: string,
  key: K,
): TextLayerStyle[K] | undefined {
  if (!layerId || pageKey !== state.currentPageKey()) return undefined;
  const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
  if (!layer) return undefined;
  return state.pendingEdits[layerId]?.estilo?.[key] ?? layer.estilo[key];
}

export function useFieldEditSession<K extends keyof TextLayerStyle>(
  layerId: string,
  key: K,
): {
  value: TextLayerStyle[K] | undefined;
  onFocus: () => void;
  onChange: (next: TextLayerStyle[K]) => void;
  onBlur: () => void;
  onCancel: () => void;
} {
  const pageKey = useEditorStore((state) => state.currentPageKey());
  const value = useEditorStore((state) => selectFieldEditValue(state, pageKey, layerId, key));
  const setWorkingEstiloPatch = useEditorStore((state) => state.setWorkingEstiloPatch);
  const recordEditorCommand = useEditorStore((state) => state.recordEditorCommand);
  const beforeRef = useRef<TextLayerStyle[K] | undefined>(undefined);
  const activeRef = useRef(false);

  const begin = useCallback(() => {
    if (activeRef.current) return;
    const currentLayer = useEditorStore.getState().getLayer(pageKey, layerId);
    beforeRef.current = currentLayer?.estilo[key];
    activeRef.current = true;
  }, [key, layerId, pageKey]);

  const finish = useCallback(() => {
    if (!activeRef.current) return;
    const before = beforeRef.current;
    const currentLayer = useEditorStore.getState().getLayer(pageKey, layerId);
    const after = currentLayer?.estilo[key];
    activeRef.current = false;
    beforeRef.current = undefined;
    recordEditorCommand({
      commandId: commandId(`edit-${String(key)}`),
      pageKey,
      createdAt: Date.now(),
      type: "edit-estilo",
      layerId,
      before: { [key]: before } as Partial<TextLayerStyle>,
      after: { [key]: after } as Partial<TextLayerStyle>,
      touchedKeys: [key],
    });
  }, [key, layerId, pageKey, recordEditorCommand]);

  return {
    value,
    onFocus: begin,
    onChange: (next) => {
      begin();
      setWorkingEstiloPatch(pageKey, layerId, { [key]: next } as Partial<TextLayerStyle>, [key]);
    },
    onBlur: finish,
    onCancel: () => {
      if (!activeRef.current) return;
      const before = beforeRef.current;
      activeRef.current = false;
      beforeRef.current = undefined;
      setWorkingEstiloPatch(pageKey, layerId, { [key]: before } as Partial<TextLayerStyle>, [key]);
    },
  };
}
