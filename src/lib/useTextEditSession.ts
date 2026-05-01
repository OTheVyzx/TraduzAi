import { useCallback, useRef } from "react";
import { useEditorStore } from "./stores/editorStore";

function commandId(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`;
}

type EditorStoreState = ReturnType<typeof useEditorStore.getState>;

export function selectTextEditValue(state: EditorStoreState, pageKey: string, layerId: string): string {
  if (!layerId || pageKey !== state.currentPageKey()) return "";
  const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
  if (!layer) return "";
  const edit = state.pendingEdits[layerId];
  return edit?.traduzido ?? edit?.translated ?? layer.traduzido ?? layer.translated ?? "";
}

export function useTextEditSession(layerId: string) {
  const pageKey = useEditorStore((state) => state.currentPageKey());
  const value = useEditorStore((state) => selectTextEditValue(state, pageKey, layerId));
  const setWorkingTraduzido = useEditorStore((state) => state.setWorkingTraduzido);
  const recordEditorCommand = useEditorStore((state) => state.recordEditorCommand);
  const beforeRef = useRef<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const composingRef = useRef(false);

  const begin = useCallback(() => {
    if (sessionIdRef.current) return;
    const currentLayer = useEditorStore.getState().getLayer(pageKey, layerId);
    beforeRef.current = currentLayer?.traduzido ?? currentLayer?.translated ?? "";
    sessionIdRef.current = crypto.randomUUID();
  }, [layerId, pageKey]);

  const finish = useCallback(() => {
    if (!sessionIdRef.current || beforeRef.current === null || composingRef.current) return;
    const currentLayer = useEditorStore.getState().getLayer(pageKey, layerId);
    const after = currentLayer?.traduzido ?? currentLayer?.translated ?? "";
    const before = beforeRef.current;
    const sessionId = sessionIdRef.current;
    beforeRef.current = null;
    sessionIdRef.current = null;
    recordEditorCommand({
      commandId: commandId("edit-texto"),
      pageKey,
      createdAt: Date.now(),
      type: "edit-traduzido",
      layerId,
      before,
      after,
      sessionId,
    });
  }, [layerId, pageKey, recordEditorCommand]);

  return {
    value,
    onFocus: begin,
    onChange: (next: string) => {
      begin();
      setWorkingTraduzido(pageKey, layerId, next);
    },
    onBlur: finish,
    onCompositionStart: () => {
      composingRef.current = true;
    },
    onCompositionEnd: () => {
      composingRef.current = false;
    },
    onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Escape") {
        event.currentTarget.blur();
      }
    },
  };
}
