import { useAppStore, type Project } from "../stores/appStore";
import { useEditorStore } from "../stores/editorStore";
import originalImageUrl from "../../../e2e/fixtures/images/page-001-original.png";
import inpaintImageUrl from "../../../e2e/fixtures/images/page-001-inpaint.png";

function buildFixtureProject(): Project {
  const textLayer = {
    id: "fixture-text-1",
    kind: "text" as const,
    source_bbox: [118, 166, 302, 252] as [number, number, number, number],
    layout_bbox: [118, 166, 302, 252] as [number, number, number, number],
    render_bbox: null,
    bbox: [118, 166, 302, 252] as [number, number, number, number],
    tipo: "fala" as const,
    original: "BURNED",
    traduzido: "TEXTO LIMPO",
    translated: "TEXTO LIMPO",
    confianca_ocr: 0.98,
    ocr_confidence: 0.98,
    estilo: {
      fonte: "ComicNeue-Bold.ttf",
      tamanho: 28,
      cor: "#ffffff",
      cor_gradiente: [],
      contorno: "#111111",
      contorno_px: 2,
      glow: false,
      glow_cor: "",
      glow_px: 0,
      sombra: false,
      sombra_cor: "",
      sombra_offset: [0, 0] as [number, number],
      bold: true,
      italico: false,
      rotacao: 0,
      alinhamento: "center" as const,
      force_upper: false,
    },
    visible: true,
    locked: false,
    order: 0,
    render_preview_path: null,
    detector: null,
    line_polygons: null,
    source_direction: null,
    rendered_direction: null,
    source_language: null,
    rotation_deg: 0,
    detected_font_size_px: null,
    balloon_bbox: [118, 166, 302, 252] as [number, number, number, number],
    balloon_subregions: [],
    layout_group_size: 1,
    qa_flags: ["visual_text_leak"],
    qa_actions: [],
  };

  return {
    id: "fixture-project",
    obra: "Fixture E2E",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
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
    paginas: [
      {
        numero: 1,
        arquivo_original: originalImageUrl,
        arquivo_traduzido: "/e2e-rendered-stale.png",
        image_layers: {
          base: { key: "base", path: originalImageUrl, visible: true, locked: true },
          inpaint: { key: "inpaint", path: inpaintImageUrl, visible: true, locked: true },
          rendered: { key: "rendered", path: "/e2e-rendered-stale.png", visible: true, locked: true },
        },
        inpaint_blocks: [],
        text_layers: [textLayer],
        textos: [textLayer],
      },
    ],
    status: "done",
    source_path: "e2e/project-basic.json",
    output_path: "e2e/project-basic.json",
    totalPages: 1,
    mode: "manual",
  };
}

let fixtureProject = buildFixtureProject();

export function getE2EFixtureProject() {
  return fixtureProject;
}

export function setE2EFixtureProject(project: Project) {
  fixtureProject = project;
}

export function resetE2EFixtureProject() {
  fixtureProject = buildFixtureProject();
  useAppStore.getState().setProject(fixtureProject);
  useEditorStore.getState().resetEditor();
  useEditorStore.setState({
    currentPageIndex: 0,
    currentPage: fixtureProject.paginas[0],
    viewMode: "translated",
    zoom: 1,
    panOffset: { x: 0, y: 0 },
  });
}

export function installE2EFixtureProject() {
  resetE2EFixtureProject();
  window.localStorage.setItem("traduzai_e2e_loaded", "1");
}
