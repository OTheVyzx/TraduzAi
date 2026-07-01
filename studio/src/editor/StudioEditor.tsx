import { useEffect, useMemo, useState } from "react";
import { Eye, EyeOff, Image, Layers, MousePointer2, Save, Type } from "lucide-react";
import { finalImagePathForPage } from "../project/adapters";
import type { ImageLayerKey, StudioPage, StudioTextLayer } from "../project/studioProject";
import { useStudioProjectStore } from "../store/projectStore";
import { bboxToPercentStyle, inferPageSize, readableLayerLabel } from "./pageGeometry";

const IMAGE_LAYER_LABELS: Partial<Record<ImageLayerKey, string>> = {
  base: "Original",
  inpaint: "Limpa",
  brush: "Brush",
  mask: "Mascara",
  rendered: "Final",
};

function imagePathForMode(page: StudioPage, mode: "original" | "clean" | "final") {
  if (mode === "original") return page.image_layers.base?.path ?? page.arquivo_original ?? finalImagePathForPage(page);
  if (mode === "clean") return page.image_layers.inpaint?.path ?? page.image_layers.base?.path ?? page.arquivo_original;
  return finalImagePathForPage(page);
}

function useCurrentPage() {
  const project = useStudioProjectStore((state) => state.project);
  const currentPageIndex = useStudioProjectStore((state) => state.currentPageIndex);
  return project?.paginas[currentPageIndex] ?? null;
}

export function StudioEditor() {
  const project = useStudioProjectStore((state) => state.project);
  const currentPageIndex = useStudioProjectStore((state) => state.currentPageIndex);
  const setCurrentPageIndex = useStudioProjectStore((state) => state.setCurrentPageIndex);
  const patchTextLayer = useStudioProjectStore((state) => state.patchCurrentTextLayer);
  const setTextLayerVisibility = useStudioProjectStore((state) => state.setCurrentTextLayerVisibility);
  const setImageLayerVisibility = useStudioProjectStore((state) => state.setCurrentImageLayerVisibility);
  const page = useCurrentPage();
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"original" | "clean" | "final">("final");
  const [draftText, setDraftText] = useState("");

  const pageSize = useMemo(() => inferPageSize(page), [page]);
  const selectedLayer = page?.text_layers.find((layer) => layer.id === selectedLayerId) ?? null;
  const imagePath = page ? imagePathForMode(page, viewMode) : null;

  useEffect(() => {
    if (!page) {
      setSelectedLayerId(null);
      return;
    }
    if (selectedLayerId && page.text_layers.some((layer) => layer.id === selectedLayerId)) return;
    setSelectedLayerId(page.text_layers[0]?.id ?? null);
  }, [page, selectedLayerId]);

  useEffect(() => {
    setDraftText(selectedLayer?.translated ?? selectedLayer?.traduzido ?? "");
  }, [selectedLayer?.id, selectedLayer?.translated, selectedLayer?.traduzido]);

  const saveSelectedLayer = async () => {
    if (!selectedLayer) return;
    await patchTextLayer(selectedLayer.id, { translated: draftText, traduzido: draftText } as Partial<StudioTextLayer>);
  };

  if (!project || !page) return null;

  return (
    <section className="studio-editor" aria-label="Editor TraduzAI Studio">
      <aside className="studio-rail" aria-label="Paginas">
        {project.paginas.map((item, index) => (
          <button
            key={`${item.numero}-${index}`}
            type="button"
            className={index === currentPageIndex ? "selected" : ""}
            onClick={() => setCurrentPageIndex(index)}
          >
            {String(item.numero).padStart(3, "0")}
          </button>
        ))}
      </aside>

      <div className="studio-workbench">
        <header className="studio-toolbar">
          <div className="tool-group" aria-label="Ferramentas">
            <button type="button" className="selected" title="Selecionar">
              <MousePointer2 size={16} />
            </button>
            <button type="button" title="Texto">
              <Type size={16} />
            </button>
            <button type="button" title="Imagem">
              <Image size={16} />
            </button>
          </div>
          <div className="segmented" aria-label="Visualizacao">
            <button type="button" className={viewMode === "original" ? "selected" : ""} onClick={() => setViewMode("original")}>
              Original
            </button>
            <button type="button" className={viewMode === "clean" ? "selected" : ""} onClick={() => setViewMode("clean")}>
              Limpa
            </button>
            <button type="button" className={viewMode === "final" ? "selected" : ""} onClick={() => setViewMode("final")}>
              Final
            </button>
          </div>
        </header>

        <div className="studio-canvas-wrap">
          <div className="studio-page" style={{ aspectRatio: `${pageSize.width} / ${pageSize.height}` }}>
            {imagePath ? (
              <img src={imagePath} alt={`Pagina ${page.numero}`} draggable={false} />
            ) : (
              <div className="studio-page-empty">Imagem indisponivel</div>
            )}
            {page.text_layers.map((layer, index) => {
              if (layer.visible === false) return null;
              return (
                <button
                  key={layer.id}
                  type="button"
                  className={`text-box ${layer.id === selectedLayerId ? "selected" : ""}`}
                  style={bboxToPercentStyle(layer.bbox, pageSize)}
                  onClick={() => setSelectedLayerId(layer.id)}
                  title={readableLayerLabel(index)}
                >
                  {layer.translated || layer.traduzido || layer.original}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <aside className="studio-inspector" aria-label="Camadas e propriedades">
        <div className="inspector-section">
          <h2><Layers size={16} /> Camadas</h2>
          <div className="layer-list">
            {Object.entries(page.image_layers).map(([key, layer]) => {
              const layerKey = key as ImageLayerKey;
              if (!layer || layer.technical) return null;
              return (
                <button key={key} type="button" onClick={() => void setImageLayerVisibility(layerKey, !layer.visible)}>
                  {layer.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                  <span>{IMAGE_LAYER_LABELS[layerKey] ?? key}</span>
                </button>
              );
            })}
            {page.text_layers.map((layer, index) => (
              <div
                key={layer.id}
                className={`layer-row ${layer.id === selectedLayerId ? "selected" : ""}`}
              >
                <button
                  type="button"
                  className="icon-inline"
                  onClick={() => void setTextLayerVisibility(layer.id, !(layer.visible ?? true))}
                  title={layer.visible === false ? "Mostrar" : "Ocultar"}
                >
                  {layer.visible === false ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
                <button type="button" className="layer-select" onClick={() => setSelectedLayerId(layer.id)}>
                  {readableLayerLabel(index)}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="inspector-section">
          <h2><Type size={16} /> Texto</h2>
          {selectedLayer ? (
            <div className="property-stack">
              <label>
                Original
                <input value={selectedLayer.original} readOnly />
              </label>
              <label>
                Traducao
                <textarea value={draftText} onChange={(event) => setDraftText(event.target.value)} rows={5} />
              </label>
              <button type="button" className="primary-action" onClick={() => void saveSelectedLayer()}>
                <Save size={16} />
                Salvar texto
              </button>
            </div>
          ) : (
            <p className="muted">Nenhum texto selecionado.</p>
          )}
        </div>
      </aside>
    </section>
  );
}
