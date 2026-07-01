import { useEffect, useMemo, useState } from "react";
import { MemoryRouter } from "react-router-dom";
import { FileDown } from "lucide-react";
import { Editor } from "../../../src/pages/Editor";
import { useAppStore, useEditorStore, type Project, type TextLayerStyle } from "../../../src/editor-shared";
import { createLegacyEditorBackendAdapter } from "../backend/editorBackendCompat";
import { getStudioEditorBackend } from "../backend/editorBackend";
import { downloadStudioPagePsd } from "../export/psd";
import type { StudioProject } from "../project/studioProject";
import { configureEditorBackend, type EditorBackendApi } from "../shims/currentEditorBackend";

const DEFAULT_TEXT_STYLE: TextLayerStyle = {
  fonte: "Comic Neue",
  tamanho: 34,
  cor: "#111827",
  cor_gradiente: [],
  contorno: "#ffffff",
  contorno_px: 2,
  glow: false,
  glow_cor: "#ffffff",
  glow_px: 0,
  sombra: false,
  sombra_cor: "#000000",
  sombra_offset: [0, 0],
  bold: true,
  italico: false,
  rotacao: 0,
  alinhamento: "center",
};

function canonicalizeTextStyle(style: unknown): TextLayerStyle {
  const record = typeof style === "object" && style !== null ? (style as Record<string, unknown>) : {};
  return {
    ...DEFAULT_TEXT_STYLE,
    ...record,
    fonte: String(record.fonte ?? record.fontFamily ?? DEFAULT_TEXT_STYLE.fonte),
    tamanho: Number(record.tamanho ?? record.fontSize ?? DEFAULT_TEXT_STYLE.tamanho),
    cor: String(record.cor ?? record.color ?? DEFAULT_TEXT_STYLE.cor),
    contorno: String(record.contorno ?? record.strokeColor ?? DEFAULT_TEXT_STYLE.contorno),
    contorno_px: Number(record.contorno_px ?? record.strokeWidth ?? DEFAULT_TEXT_STYLE.contorno_px),
    rotacao: Number(record.rotacao ?? record.rotation ?? DEFAULT_TEXT_STYLE.rotacao),
    alinhamento:
      record.alinhamento === "left" || record.alinhamento === "right" || record.alinhamento === "center"
        ? record.alinhamento
        : DEFAULT_TEXT_STYLE.alinhamento,
  };
}

function repairMaskLeakedIntoInpaint(page: StudioProject["paginas"][number]) {
  const maskPath = page.image_layers.mask?.path;
  const inpaintPath = page.image_layers.inpaint?.path;
  const fallbackPath = page.image_layers.base?.path ?? page.arquivo_original ?? page.arquivo_traduzido ?? null;
  if (!maskPath || !inpaintPath || inpaintPath !== maskPath || !fallbackPath) return page;
  return {
    ...page,
    image_layers: {
      ...page.image_layers,
      inpaint: {
        ...(page.image_layers.inpaint ?? {}),
        key: "inpaint" as const,
        path: fallbackPath,
        visible: true,
      },
    },
  };
}

function toAppProject(project: StudioProject, projectPath: string): Project {
  const paginas = project.paginas.map((inputPage) => {
    const page = repairMaskLeakedIntoInpaint(inputPage);
    const textLayers = page.text_layers.map((layer, index) => {
      const estilo = canonicalizeTextStyle(layer.estilo ?? layer.style);
      return {
        ...layer,
        bbox: layer.bbox,
        layout_bbox: layer.layout_bbox ?? layer.bbox,
        original: layer.original ?? "",
        traduzido: layer.traduzido ?? layer.translated ?? "",
        translated: layer.translated ?? layer.traduzido ?? "",
        tipo: layer.tipo ?? "fala",
        confianca_ocr: layer.confianca_ocr ?? layer.ocr_confidence ?? 1,
        ocr_confidence: layer.ocr_confidence ?? layer.confianca_ocr ?? 1,
        visible: layer.visible !== false,
        locked: layer.locked === true,
        order: layer.order ?? index,
        estilo,
        style: estilo,
      };
    });
    return {
      ...page,
      arquivo_original: page.arquivo_original ?? page.image_layers.base?.path ?? "",
      arquivo_traduzido: page.arquivo_traduzido ?? page.image_layers.rendered?.path ?? "",
      text_layers: textLayers,
      textos: textLayers,
    };
  }) as Project["paginas"];
  return {
    ...(project as unknown as Record<string, unknown>),
    id: project.id ?? "traduzai-studio-project",
    obra: project.obra ?? "TraduzAI Studio",
    capitulo: Number(project.capitulo ?? 1),
    idioma_origem: project.idioma_origem ?? "en",
    idioma_destino: project.idioma_destino ?? "pt-BR",
    qualidade: "normal",
    contexto: {
      sinopse: "",
      genero: [],
      personagens: [],
      glossario: {},
      aliases: [],
      termos: [],
      relacoes: [],
      faccoes: [],
      resumo_por_arco: [],
      memoria_lexical: {},
      fontes_usadas: [],
    },
    paginas,
    status: "done",
    source_path: projectPath,
    output_path: projectPath,
    totalPages: project.paginas.length,
    mode: "manual",
  };
}

export function StudioSharedEditor({
  project,
  projectPath,
}: {
  project: StudioProject;
  projectPath: string;
}) {
  const appProject = useMemo(() => toAppProject(project, projectPath), [project, projectPath]);
  const resetEditor = useEditorStore((state) => state.resetEditor);
  const currentPageIndex = useEditorStore((state) => state.currentPageIndex);
  const commitEdits = useEditorStore((state) => state.commitEdits);
  const [isExportingPsd, setIsExportingPsd] = useState(false);

  useEffect(() => {
    const backend = createLegacyEditorBackendAdapter(getStudioEditorBackend()) as unknown as EditorBackendApi;
    configureEditorBackend(backend);
    useAppStore.getState().setProject(appProject);
    resetEditor();
  }, [appProject, resetEditor]);

  const currentPage = project.paginas[currentPageIndex] ?? project.paginas[0] ?? null;

  const exportCurrentPagePsd = async () => {
    setIsExportingPsd(true);
    try {
      await commitEdits();
      const latestProject = await getStudioEditorBackend().loadProject({ project_path: projectPath });
      await downloadStudioPagePsd(latestProject, currentPageIndex);
    } catch (error) {
      console.error("Falha ao exportar PSD no Studio:", error);
    } finally {
      setIsExportingPsd(false);
    }
  };

  return (
    <MemoryRouter>
      <Editor
        onBack={() => undefined}
        emptyBackLabel="Voltar ao Studio"
        headerActions={
          <button
            type="button"
            onClick={() => void exportCurrentPagePsd()}
            disabled={isExportingPsd || !currentPage}
            className="flex items-center gap-1 rounded-lg border border-status-success/30 bg-status-success/10 px-2.5 py-1 text-[11px] font-medium text-status-success transition-smooth hover:bg-status-success/15 disabled:opacity-30"
            title="Salvar pagina atual em PSD"
          >
            <FileDown size={12} />
            Salvar em PSD
          </button>
        }
      />
    </MemoryRouter>
  );
}
