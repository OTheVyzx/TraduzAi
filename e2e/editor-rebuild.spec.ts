import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const criticalPatterns = [
  /getSnapshot should be cached/i,
  /Maximum update depth exceeded/i,
  /React fatal/i,
  /Zustand fatal/i,
];

const realGeneratedProjectPath = process.env.TRADUZAI_E2E_REAL_PROJECT
  ? path.resolve(process.env.TRADUZAI_E2E_REAL_PROJECT)
  : path.resolve("debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/work/project.json");
const e2eCompletionCoverUrl =
  "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20120%20180%22%3E%3Crect%20width%3D%22120%22%20height%3D%22180%22%20fill%3D%22%236c5ce7%22%2F%3E%3Ctext%20x%3D%2260%22%20y%3D%2294%22%20fill%3D%22white%22%20font-size%3D%2218%22%20text-anchor%3D%22middle%22%3ECAPA%3C%2Ftext%3E%3C%2Fsvg%3E";

function loadRealGeneratedProjectForE2E() {
  const raw = JSON.parse(readFileSync(realGeneratedProjectPath, "utf8"));
  return {
    ...raw,
    id: "real-generated-project-e2e",
    qualidade: "normal",
    contexto: raw.contexto ?? {
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
    paginas: raw.paginas ?? [],
    status: "done",
    source_path: realGeneratedProjectPath,
    output_path: realGeneratedProjectPath,
    totalPages: raw.paginas?.length ?? 0,
    mode: "auto",
  };
}

function buildTwoPageEditorReaderProject() {
  const makeImage = (label: string, fill: string) =>
    `data:image/svg+xml,${encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 640">
        <rect width="420" height="640" fill="${fill}"/>
        <rect x="32" y="32" width="356" height="576" fill="#fff" stroke="#111" stroke-width="6"/>
        <text x="210" y="320" fill="#111" font-size="52" font-family="Arial" text-anchor="middle">${label}</text>
      </svg>`,
    )}`;
  const pageOneImage = makeImage("P1", "#f7f7fb");
  const pageTwoImage = makeImage("P2", "#eef8ff");
  const pageThreeImage = makeImage("P3", "#fff8ed");
  const style = {
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
    sombra_offset: [0, 0],
    bold: true,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
    force_upper: false,
  };
  const makeLayer = (id: string, text: string) => ({
    id,
    kind: "text",
    source_bbox: [118, 166, 302, 252],
    layout_bbox: [118, 166, 302, 252],
    render_bbox: null,
    bbox: [118, 166, 302, 252],
    tipo: "fala",
    original: text,
    traduzido: text,
    translated: text,
    confianca_ocr: 0.98,
    ocr_confidence: 0.98,
    estilo: style,
    style,
    visible: true,
    locked: false,
    order: 0,
  });

  return {
    id: "editor-reader-scroll-e2e",
    obra: "Editor Reader Scroll E2E",
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
        arquivo_original: pageOneImage,
        arquivo_traduzido: pageOneImage,
        image_layers: {
          base: { key: "base", path: pageOneImage, visible: true, locked: true },
          inpaint: { key: "inpaint", path: pageOneImage, visible: true, locked: true },
          rendered: { key: "rendered", path: pageOneImage, visible: true, locked: true },
        },
        text_layers: [makeLayer("reader-page-1", "PAGINA UM")],
      },
      {
        numero: 2,
        arquivo_original: pageTwoImage,
        arquivo_traduzido: pageTwoImage,
        image_layers: {
          base: { key: "base", path: pageTwoImage, visible: true, locked: true },
          inpaint: { key: "inpaint", path: pageTwoImage, visible: true, locked: true },
          rendered: { key: "rendered", path: pageTwoImage, visible: true, locked: true },
        },
        text_layers: [makeLayer("reader-page-2", "PAGINA DOIS")],
      },
      {
        numero: 3,
        arquivo_original: pageThreeImage,
        arquivo_traduzido: pageThreeImage,
        image_layers: {
          base: { key: "base", path: pageThreeImage, visible: true, locked: true },
          inpaint: { key: "inpaint", path: pageThreeImage, visible: true, locked: true },
          rendered: { key: "rendered", path: pageThreeImage, visible: true, locked: true },
        },
        text_layers: [makeLayer("reader-page-3", "PAGINA TRES")],
      },
    ],
    status: "done",
    source_path: "e2e/project-reader-scroll.json",
    output_path: "e2e/project-reader-scroll.json",
    totalPages: 3,
    mode: "manual",
  };
}

test("editor Konva usa fundo limpo e layers editaveis @smoke", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/editor");

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 15000 });
  await expect(stage).toHaveAttribute("data-base-kind", "inpaint");
  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByTestId("editor-view-original")).toBeVisible();
  await expect(page.getByTitle("Selecionar (V)")).toBeVisible();
  await expect(page.getByTitle("Novo bloco de texto (T)")).toBeVisible();
  await expect(page.getByTitle("Brush (B)")).toBeVisible();
  await expect(page.getByTitle("Borracha (E)")).toBeVisible();
  await expect(page.getByTitle(/Lasso/)).toBeVisible();

  await expect(page.getByText("Fixture E2E")).toBeVisible();
  await expect(page.getByText("TEXTO LIMPO")).toBeVisible();
  await expect(page.getByText("BURNED")).toHaveCount(0);

  async function layerState() {
    const raw = await page.getByTestId("editor-stage-state").getAttribute("data-layers");
    const layers = JSON.parse(raw ?? "[]") as Array<{
      id: string;
      bbox: [number, number, number, number];
      visible: boolean;
      locked: boolean;
      text: string;
      color: string;
      rotation: number;
    }>;
    return layers[0];
  }

  const canvas = stage.locator("canvas").first();
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) throw new Error("Canvas sem bounding box");

  await page.getByText("TEXTO LIMPO").click();
  const sizeInput = page.getByTitle("Tamanho da fonte");
  await expect(sizeInput).toBeVisible();
  await expect(sizeInput).toHaveValue("28");

  await sizeInput.fill("40");
  await expect(sizeInput).toHaveValue("40");
  await page.getByTitle(/Desfazer: Editar estilo/).click();
  await expect(sizeInput).toHaveValue("28");
  await page.getByTitle(/Refazer: Editar estilo/).click();
  await expect(sizeInput).toHaveValue("40");

  const rotationInput = page.getByRole("spinbutton", { name: "Rotacao" });
  await expect(rotationInput).toBeVisible();
  await expect(rotationInput).toHaveValue("0");
  await rotationInput.fill("15");
  await expect.poll(async () => (await layerState()).rotation).toBe(15);
  await page.getByTitle("Zerar rotacao").click();
  await expect.poll(async () => (await layerState()).rotation).toBe(0);
  await page.getByTitle("Girar +15 graus").click();
  await expect.poll(async () => (await layerState()).rotation).toBe(15);
  await page.getByTitle("Zerar rotacao").click();
  await expect.poll(async () => (await layerState()).rotation).toBe(0);

  const rotateLayer = await layerState();
  const canvasSize = await canvas.evaluate((node: HTMLCanvasElement) => ({
    width: node.width,
    height: node.height,
  }));
  const refreshedBox = await canvas.boundingBox();
  expect(refreshedBox).not.toBeNull();
  if (!refreshedBox) throw new Error("Canvas sem bounding box para rotacao");
  const toViewportPoint = (x: number, y: number) => ({
    x: refreshedBox.x + (x / canvasSize.width) * refreshedBox.width,
    y: refreshedBox.y + (y / canvasSize.height) * refreshedBox.height,
  });
  const [rx1, ry1, rx2, ry2] = rotateLayer.bbox;
  const rotateStart = toViewportPoint(rx2 + 12, (ry1 + ry2) / 2);
  const rotateEnd = toViewportPoint((rx1 + rx2) / 2, ry1 - 80);
  await page.mouse.move(rotateStart.x, rotateStart.y);
  await page.mouse.down();
  await page.mouse.move(rotateEnd.x, rotateEnd.y, { steps: 10 });
  await page.mouse.up();
  await expect.poll(async () => Math.abs((await layerState()).rotation)).toBeGreaterThan(10);

  await page.getByTitle("Cor do texto").fill("#ff0000");
  await expect.poll(async () => (await layerState()).color).toBe("#ff0000");

  await page.getByTitle("Ocultar camada de texto").click();
  await expect.poll(async () => (await layerState()).visible).toBe(false);
  await page.getByTitle("Mostrar camada de texto").click();
  await expect.poll(async () => (await layerState()).visible).toBe(true);

  await page.getByTitle("Bloquear camada de texto").click();
  await expect.poll(async () => (await layerState()).locked).toBe(true);
  await page.getByTitle("Desbloquear camada de texto").click();

  await page.getByTitle("Novo bloco de texto (T)").click();
  const beforeCreateCount = JSON.parse((await page.getByTestId("editor-stage-state").getAttribute("data-layers")) ?? "[]").length;
  await page.mouse.move(box.x + box.width * 0.18, box.y + box.height * 0.58);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.36, box.y + box.height * 0.70, { steps: 8 });
  await page.mouse.up();
  await expect
    .poll(async () => JSON.parse((await page.getByTestId("editor-stage-state").getAttribute("data-layers")) ?? "[]").length)
    .toBe(beforeCreateCount + 1);

  await page.getByTitle("Brush (B)").click();
  await page.mouse.move(box.x + box.width * 0.20, box.y + box.height * 0.22);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.30, box.y + box.height * 0.28, { steps: 5 });
  await page.mouse.up();
  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByTitle(/Desfazer: Pincel/)).toBeEnabled();
  await page.keyboard.press("Control+Z");
  await expect(page.getByTitle(/Refazer: Pincel/)).toBeEnabled();
  await page.keyboard.press("Control+Y");
  await expect(page.getByTitle(/Desfazer: Pincel/)).toBeEnabled();

  await page.getByTitle(/Lasso/).click();
  await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.22);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.28, { steps: 5 });
  await page.mouse.up();
  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-selection", /"points"/);
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-selection", /"bbox"/);
  await expect(page.getByTestId("lasso-context-menu")).toBeVisible();
  await expect(page.getByTitle("OCR area")).toBeVisible();
  const lassoSelection = await page.getByTestId("editor-stage-state").getAttribute("data-lasso-selection");
  expect(lassoSelection).toBeTruthy();
  await page.mouse.click(box.x + box.width * 0.48, box.y + box.height * 0.25);
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-selection", lassoSelection ?? "");
  await page.getByTitle("Cancelar seleção").click();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-selection", "");
  await page.mouse.move(box.x + box.width * 0.24, box.y + box.height * 0.36);
  await page.mouse.down();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-progress-points", "1");
  await page.mouse.up();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-progress-points", "0");
  await page.mouse.move(box.x + box.width * 0.56, box.y + box.height * 0.44, { steps: 5 });
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-lasso-progress-points", "0");

  const criticalErrors = errors.filter((message) => criticalPatterns.some((pattern) => pattern.test(message)));
  expect(criticalErrors).toEqual([]);
});

test("editor aplica preset visual ao texto selecionado @text-presets", async ({ page }) => {
  await page.goto("/editor");

  await expect(page.getByTestId("editor-stage")).toBeVisible({ timeout: 15000 });
  await page.getByText("TEXTO LIMPO").click();

  const layerState = async () => {
    const raw = await page.getByTestId("editor-stage-state").getAttribute("data-layers");
    const layers = JSON.parse(raw ?? "[]") as Array<{ color: string; gradient: string[]; rotation: number }>;
    return layers[0];
  };

  await page.getByTestId("text-style-preset-button").click();
  await expect(page.getByTestId("text-style-preset-popover")).toBeVisible();
  await page.getByTestId("text-style-preset-option-bang_comic").click();

  await expect(page.getByTestId("text-font-select")).toHaveValue("KOMIKAX_.ttf");
  await expect.poll(async () => (await layerState()).color).toBe("#ffe900");
  await expect.poll(async () => (await layerState()).gradient).toEqual(["#fff247", "#ff7a00"]);
  await expect.poll(async () => (await layerState()).rotation).toBe(0);

  await page.getByTitle("Cor do texto").fill("#ff0000");
  await expect.poll(async () => (await layerState()).color).toBe("#ff0000");
  await expect.poll(async () => (await layerState()).gradient).toEqual([]);

  await page.getByTitle("Gradiente").click();
  await expect(page.getByTestId("text-gradient-preview")).toBeVisible();
  await page.getByTestId("text-gradient-end").fill("#00ffff");
  await expect.poll(async () => (await layerState()).gradient).toEqual(["#ff0000", "#00ffff"]);
  await page.getByTestId("text-gradient-clear").click();
  await expect.poll(async () => (await layerState()).gradient).toEqual([]);
});

test("editor abre project.json gerado real @real-project", async ({ page }) => {
  test.skip(!existsSync(realGeneratedProjectPath), `Projeto real nao encontrado: ${realGeneratedProjectPath}`);

  const project = loadRealGeneratedProjectForE2E();
  expect(project.paginas.length).toBeGreaterThan(1);
  expect(project.paginas[0].text_layers.length).toBeGreaterThan(1);

  await page.addInitScript((injectedProject) => {
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
  }, project);

  await page.goto("/editor");

  await expect(page.getByTestId("editor-stage")).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("Grand Finale")).toBeVisible();
  await expect(page.getByText(`1/${project.paginas.length}`)).toBeVisible();

  await expect
    .poll(async () => {
      const rawLayers = await page.getByTestId("editor-stage-state").getAttribute("data-layers");
      return JSON.parse(rawLayers ?? "[]").length;
    })
    .toBe(project.paginas[0].text_layers.length);

  const rawLayers = await page.getByTestId("editor-stage-state").getAttribute("data-layers");
  const layers = JSON.parse(rawLayers ?? "[]") as Array<{ id: string; text: string }>;
  expect(layers.every((layer) => layer.id.length > 0)).toBe(true);
  expect(layers.some((layer) => /Todos os quadrinhos/i.test(layer.text))).toBe(true);
});

test("editor mantem imagem editavel visivel depois de salvar @smoke", async ({ page }) => {
  await page.goto("/editor");

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 15000 });
  await expect(stage).toHaveAttribute("data-text-editing", "true");

  await page.getByText("TEXTO LIMPO").click();
  const sizeInput = page.getByTitle("Tamanho da fonte");
  await sizeInput.fill("36");

  const saveButton = page.getByRole("button", { name: "Salvar", exact: true });
  await expect(saveButton).toBeEnabled();
  await saveButton.click();

  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByText("Carregando imagem...")).toHaveCount(0);
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /TEXTO LIMPO/);
});

test("editor usa leitor vertical com barreira e sincroniza textos por scroll e miniatura @editor", async ({ page }) => {
  await page.addInitScript((injectedProject) => {
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
  }, buildTwoPageEditorReaderProject());

  await page.goto("/editor", { waitUntil: "domcontentloaded" });

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("1/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA UM/);

  const viewport = page.getByTestId("editor-reader-viewport");
  const firstPage = page.getByTestId("editor-reader-page-1");
  const secondPage = page.getByTestId("editor-reader-page-2");
  const barrier = page.getByTestId("editor-reader-barrier-1-2");
  await expect(viewport).toBeVisible();
  await expect(firstPage).toBeVisible();
  await expect(secondPage).toBeVisible();
  await expect(barrier).toBeVisible();
  await expect(page.getByTestId("editor-page-turn-preview")).toHaveCount(0);

  const firstBox = await firstPage.boundingBox();
  const secondBox = await secondPage.boundingBox();
  expect(firstBox).not.toBeNull();
  expect(secondBox).not.toBeNull();
  if (!firstBox || !secondBox) throw new Error("Paginas do leitor sem bounding box");
  expect(secondBox.y - (firstBox.y + firstBox.height)).toBeGreaterThan(56);

  await viewport.evaluate((node) => {
    const target = node.querySelector('[data-testid="editor-reader-page-2"]') as HTMLElement | null;
    if (!target) throw new Error("Pagina 2 nao encontrada");
    node.scrollTop = target.offsetTop + target.offsetHeight / 3;
    node.dispatchEvent(new Event("scroll", { bubbles: true }));
  });

  await expect(page.getByText("2/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "1");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA DOIS/);
  await page.waitForTimeout(500);
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "1");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA DOIS/);

  await page.getByTestId("editor-page-thumbnail-1").click();
  await expect(page.getByText("1/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "0");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA UM/);
  await expect
    .poll(async () =>
      viewport.evaluate((node) => {
        const target = node.querySelector('[data-testid="editor-reader-page-1"]') as HTMLElement | null;
        if (!target) return false;
        const viewportRect = node.getBoundingClientRect();
        const targetRect = target.getBoundingClientRect();
        return targetRect.bottom > viewportRect.top && targetRect.top < viewportRect.bottom;
      }),
    )
    .toBe(true);

  await page.getByTitle("Diminuir zoom (-)").click();
  await expect(page.getByText("85%")).toBeVisible();
  await page.getByTestId("editor-reader-go-down-2").click();
  await expect(page.getByText("2/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "1");
  await expect(page.getByText("PAGINA DOIS")).toBeVisible();
  await expect(page.getByText("85%")).toBeVisible();

  await page.getByTestId("editor-page-thumbnail-3").click();
  await expect(page.getByText("3/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "2");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA TRES/);
  await page.waitForTimeout(300);

  await viewport.evaluate((node) => {
    const target = node.querySelector('[data-testid="editor-reader-page-2"]') as HTMLElement | null;
    if (!target) throw new Error("Pagina 2 nao encontrada");
    node.scrollTop = target.offsetTop + target.offsetHeight / 3;
    node.dispatchEvent(new Event("scroll", { bubbles: true }));
  });

  await expect(page.getByText("2/3")).toBeVisible();
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "1");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA DOIS/);
  await page.waitForTimeout(500);
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-page-index", "1");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /PAGINA DOIS/);
});

test("pincel de recuperacao mantem textos visiveis sem alternar camadas tecnicas @editor", async ({ page }) => {
  await page.goto("/editor");

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 15000 });
  await expect(stage).toHaveAttribute("data-base-kind", "inpaint");
  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /TEXTO LIMPO/);

  const canvas = stage.locator("canvas").first();
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) throw new Error("Canvas sem bounding box");

  await page.keyboard.press("r");
  await page.mouse.move(box.x + box.width * 0.46, box.y + box.height * 0.36);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.42, { steps: 5 });
  await page.mouse.up();

  await expect(stage).toHaveAttribute("data-base-kind", "inpaint");
  await expect(stage).toHaveAttribute("data-text-editing", "true");
  await expect(page.getByTestId("editor-stage-state")).toHaveAttribute("data-layers", /TEXTO LIMPO/);
  await expect(page.getByText("Recuperacao")).toHaveCount(0);
  await expect(page.getByText("Tecnicas")).toHaveCount(0);
});

test("alt mais botao direito arrastando ajusta tamanho do pincel globalmente @editor", async ({ page }) => {
  await page.goto("/editor");

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 15000 });
  const canvas = stage.locator("canvas").first();
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) throw new Error("Canvas sem bounding box");

  await page.getByTitle("Brush (B)").click();
  const sizeInput = page.getByTitle("Tamanho do pincel");
  await expect(sizeInput).toHaveValue("18");

  await page.keyboard.down("Alt");
  await page.mouse.move(box.x + box.width * 0.50, box.y + box.height * 0.50);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(box.x + box.width * 0.66, box.y + box.height * 0.50, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await page.keyboard.up("Alt");

  await expect.poll(async () => Number(await sizeInput.inputValue())).toBeGreaterThan(18);
  const grownSize = Number(await sizeInput.inputValue());

  await page.getByTitle("Borracha (E)").click();
  await expect(page.getByTitle("Tamanho do pincel")).toHaveValue(String(grownSize));

  await page.keyboard.down("Alt");
  await page.mouse.move(box.x + box.width * 0.50, box.y + box.height * 0.50);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(box.x + box.width * 0.38, box.y + box.height * 0.50, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await page.keyboard.up("Alt");

  await expect.poll(async () => Number(await page.getByTitle("Tamanho do pincel").inputValue())).toBeLessThan(grownSize);
});

test("home mostra onboarding inicial e permite pular @onboarding", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByTestId("onboarding-modal")).toBeVisible();
  await expect(page.getByText("Como o TraduzAI funciona")).toBeVisible();
  await expect(page.getByText("Importe um capitulo")).toBeVisible();

  await page.getByTestId("skip-onboarding").click();
  await expect(page.getByTestId("onboarding-modal")).toHaveCount(0);

  await page.getByTestId("open-onboarding").click();
  await expect(page.getByTestId("onboarding-modal")).toBeVisible();
});

test("home abre projeto recente ao clicar no card @home", async ({ page }) => {
  test.skip(!existsSync(realGeneratedProjectPath), `Projeto real nao encontrado: ${realGeneratedProjectPath}`);

  const project = loadRealGeneratedProjectForE2E();
  expect(project.paginas.length).toBeGreaterThan(0);

  await page.addInitScript((injectedProject) => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
    window.localStorage.setItem(
      "traduzai_recent",
      JSON.stringify([
        {
          id: "recent-real-generated-project",
          obra: injectedProject.obra,
          capitulo: injectedProject.capitulo,
          pages: injectedProject.paginas.length,
          date: new Date(0).toISOString(),
          status: "done",
          project_path: injectedProject.output_path,
        },
      ])
    );
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
  }, project);

  await page.goto("/");

  const recentTitle = page.getByText(project.obra, { exact: true });
  await expect(recentTitle).toBeVisible();
  await recentTitle.click();

  await expect(page).toHaveURL(/\/preview$/);
  await expect(page.getByTestId("preview-page-counter")).toBeVisible({ timeout: 15000 });
});

test("layout aplica gradiente global @settings", async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1920, height: 1000 });

  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/settings", { waitUntil: "domcontentloaded" });

  await expect(page.getByTestId("app-shell")).toHaveClass(/app-gradient-shell/);
  await expect(page.getByTestId("app-sidebar")).toHaveClass(/app-sidebar-minimal/);
  const sidebarClass = await page.getByTestId("app-sidebar").getAttribute("class");
  expect(sidebarClass ?? "").not.toContain("app-sidebar-gradient");
  expect(sidebarClass ?? "").not.toContain("border-r");
  await expect(page.getByTestId("app-sidebar").locator("> .h-px")).toHaveCount(0);
  const mainClass = await page.getByTestId("app-main").getAttribute("class");
  expect(mainClass ?? "").not.toContain("app-main-gradient");
  await expect(page.getByTestId("app-main")).toHaveClass(/app-main-unified/);
  await expect(page.getByTestId("settings-appearance-section")).toBeVisible();

  const settingsBox = await page.getByTestId("settings-page").boundingBox();
  const viewport = page.viewportSize();
  expect(settingsBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  if (!settingsBox || !viewport) throw new Error("Sem dimensoes para validar centralizacao");
  const settingsCenter = settingsBox.x + settingsBox.width / 2;
  expect(Math.abs(settingsCenter - viewport.width / 2)).toBeLessThan(12);
});

test("home centraliza conteudo pela janela @home", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1000 });
  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });

  const homeBox = await page.getByTestId("home-content").boundingBox();
  const viewport = page.viewportSize();
  expect(homeBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  if (!homeBox || !viewport) throw new Error("Sem dimensoes para validar centralizacao");
  const homeCenter = homeBox.x + homeBox.width / 2;
  expect(Math.abs(homeCenter - viewport.width / 2)).toBeLessThan(12);
});

test("home fica no centro vertical da janela @home", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1000 });
  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });

  const homeBox = await page.getByTestId("home-content").boundingBox();
  expect(homeBox).not.toBeNull();
  if (!homeBox) throw new Error("Sem dimensoes para validar alinhamento vertical");
  expect(homeBox.y).toBeGreaterThan(120);
});

test("layout centraliza sem deslocar a pagina @home", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1000 });
  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });

  await expect(page.getByTestId("app-main")).toHaveCSS("padding-right", "224px");
  await expect(page.getByTestId("home-content")).toHaveCSS("left", "0px");
});

test("settings fica no centro vertical da janela @settings", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1000 });
  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/settings", { waitUntil: "domcontentloaded" });

  const settingsBox = await page.getByTestId("settings-page").boundingBox();
  const appearanceBox = await page.getByTestId("settings-appearance-section").boundingBox();
  const viewport = page.viewportSize();
  expect(settingsBox).not.toBeNull();
  expect(appearanceBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  if (!settingsBox || !appearanceBox || !viewport) {
    throw new Error("Sem dimensoes para validar alinhamento vertical");
  }
  expect(settingsBox.height).toBeGreaterThan(viewport.height - 4);
  expect(appearanceBox.y).toBeGreaterThan(160);
});

test("preview usa layout de trabalho sem sidebar @phase17", async ({ page }) => {
  test.setTimeout(60_000);

  await page.goto("/preview", { waitUntil: "domcontentloaded" });

  await expect(page.getByTestId("app-shell")).toHaveClass(/app-workspace-shell/);
  await expect(page.getByTestId("app-shell")).not.toHaveClass(/app-gradient-shell/);
  await expect(page.getByTestId("app-sidebar")).toHaveCount(0);
  await expect(page.getByTestId("app-main")).toHaveClass(/app-main-workspace/);
  await expect(page.getByTestId("preview-viewport")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("qa-panel")).toBeVisible();
});

test("settings nao expõe Ollama como opcao de traducao @settings", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
  });

  await page.goto("/settings", { waitUntil: "domcontentloaded" });

  await expect(page.locator("body")).not.toContainText("Traducao local (Ollama)");
  await expect(page.locator("body")).not.toContainText("Ollama");
  await expect(page.locator("body")).not.toContainText("LLM pronto");
  await expect(page.locator("body")).not.toContainText("Sem modelo");

  const activeConfig = page.getByRole("button", { name: /Config/ });
  await expect(activeConfig).toHaveAttribute("data-active", "true");
  const activeConfigClass = await activeConfig.getAttribute("class");
  expect(activeConfigClass ?? "").not.toContain("bg-brand/10");
  expect(activeConfigClass ?? "").not.toContain("shadow-[");

  await expect(page.getByTestId("settings-appearance-section")).toBeVisible();
  await expect(page.getByTestId("settings-theme-dark")).toBeVisible();
  await expect(page.getByTestId("settings-theme-light")).toBeVisible();
  await expect(page.getByTestId("settings-theme-system")).toBeVisible();
  await expect(page.getByTestId("settings-app-language")).toBeVisible();

  await page.getByTestId("settings-theme-light").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByTestId("settings-theme-dark").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await expect(page.getByTestId("settings-system-section")).toContainText("GPU");
  await expect(page.getByTestId("settings-system-section")).not.toContainText("Modelos OCR");
  await expect(page.getByTestId("settings-system-section")).not.toContainText("Creditos");

  await expect(page.getByTestId("settings-packages-section")).toBeVisible();
  await page.getByTestId("settings-download-packages").click();
  await expect(page.getByTestId("settings-package-log")).toContainText("Iniciando download");

  await expect(page.locator("body")).not.toContainText("Idiomas padrao");
  await expect(page.locator("body")).not.toContainText("Idioma de origem");
  await expect(page.locator("body")).not.toContainText("Idioma de destino");
});

test("traducao manual pula configuracao e abre editor @manual-flow", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("skip-onboarding").click();

  await page.getByRole("button", { name: /Tradu..o Manual/i }).click();

  await expect(page).not.toHaveURL(/\/setup$/);
  await expect(page).toHaveURL(/\/editor$/);
  await expect(page.getByTestId("editor-stage")).toBeVisible();
  await expect(page.getByText("Fixture E2E")).toBeVisible();
});

test("setup avisa quando nenhuma obra foi selecionada @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("");
  await page.getByTestId("setup-start-button").click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Escolha uma obra antes de iniciar")).toBeVisible();
  await expect(page.getByText("Informe o nome da obra para carregar contexto antes de iniciar.")).toBeVisible();
});

test("setup avisa quando obra esta sem contexto ativo @phase4 @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await expect(page.getByTestId("project-name-input")).toHaveValue("Fixture E2E");
  await page.getByTestId("setup-start-button").click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Contexto da obra nao encontrado")).toBeVisible();

  await page.getByTestId("work-context-continue-without-context").click();
  await expect(page).toHaveURL(/\/processing$/);
});

test("setup diferencia contexto carregado de glossario vazio @setup", async ({ page }) => {
  const project = loadRealGeneratedProjectForE2E();
  project.obra = "Reincarnated Murim Lord";
  project.contexto = {
    ...project.contexto,
    sinopse: "Regret is the final emotion of Grand Martial Alliance founder Hyeok Ryeon Mugang.",
    glossario: {},
    memoria_lexical: { "Murim Lord": "Lorde Murim" },
  };
  project.work_context = {
    selected: true,
    work_id: "reincarnated-murim-lord",
    title: project.obra,
    context_loaded: true,
    glossary_loaded: false,
    glossary_entries_count: 0,
    internet_context_loaded: true,
    risk_level: "medium",
    user_ignored_warning: false,
  };

  await page.addInitScript((injectedProject) => {
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
  }, project);

  await page.goto("/setup");
  await page.getByTestId("project-name-input").fill(project.obra);
  await expect(page.getByTestId("project-name-input")).toHaveValue(project.obra);

  await page.getByTestId("setup-start-button").click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Glossario da obra vazio")).toBeVisible();
  await expect(page.getByText("Contexto da obra nao encontrado")).toHaveCount(0);
});

test("setup permite adicionar e remover termo do glossario @phase5", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await page.getByTestId("setup-start-button").click();
  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await page.getByRole("button", { name: "Revisar glossario" }).click();
  await expect(page.getByTestId("setup-advanced-panel")).toHaveAttribute("open", "");

  await page.getByPlaceholder("Termo origem").fill("Knight");
  await page.getByPlaceholder("Traducao").fill("Cavaleiro");
  await page.getByTestId("glossary-add-entry-button").click();

  await expect(page.getByText("Knight")).toBeVisible();
  await expect(page.getByText("Cavaleiro")).toBeVisible();

  await page.getByTitle("Remover termo").click();
  await expect(page.getByText("Knight")).toHaveCount(0);
});

test("setup busca contexto online e mostra glossario sugerido @internet-context @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("The Regressed Mercenary Has a Plan");
  await expect(page.getByTestId("work-context-risk")).toContainText("alto");
  await expect(page.getByTestId("setup-advanced-panel")).not.toHaveAttribute("open", "");
  await page.getByTestId("project-search-button").click();
  await expect(page.getByTestId("work-result-item").first()).toBeVisible();

  await page.getByTestId("work-result-item").first().click();
  await expect(page.getByTestId("glossary-suggestions")).toBeVisible();
  await expect(page.getByTestId("glossary-suggestion-row").filter({ hasText: "Ghislain Perdium" })).toBeVisible();
  await expect(page.getByTestId("work-context-summary")).toContainText("Sugestoes: 3");

  await page.getByTestId("setup-start-button").click();
  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Revise as sugestoes de glossario")).toBeVisible();
  await page.getByRole("button", { name: "Revisar sugestoes" }).click();
  await expect(page.getByTestId("work-context-warning-modal")).toHaveCount(0);

  await page
    .getByTestId("glossary-suggestion-row")
    .filter({ hasText: "Ghislain Perdium" })
    .getByTestId("glossary-suggestion-accept")
    .click();
  await expect(page.getByTestId("work-context-summary")).toContainText("Aceitos: 1");
  await expect(page.getByTestId("work-context-risk")).toContainText("medio");
});

test("setup revisa candidatos no glossario central @glossary-center", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("The Regressed Mercenary Has a Plan");
  await page.getByTestId("project-search-button").click();
  await page.getByTestId("work-result-item").first().click();

  await expect(page.getByTestId("glossary-suggestion-row").filter({ hasText: "Ghislain Perdium" })).toBeVisible();

  await page
    .getByTestId("glossary-suggestion-row")
    .filter({ hasText: "Ghislain Perdium" })
    .getByTestId("glossary-suggestion-accept")
    .click();

  await expect(page.getByTestId("work-context-summary")).toContainText("Aceitos: 1");

  await page
    .getByTestId("glossary-suggestion-row")
    .filter({ hasText: "Cavald" })
    .getByTestId("glossary-suggestion-ignore")
    .click();

  await expect(page.getByTestId("glossary-suggestions").getByText("Cavald")).toHaveCount(0);

  await page
    .getByTestId("glossary-suggestion-row")
    .filter({ hasText: "mana technique" })
    .getByTestId("glossary-suggestion-edit")
    .click();
  await page.getByTestId("glossary-suggestion-edit-target").fill("Tecnica de mana");
  await page.getByTestId("glossary-suggestion-save-edit").click();
  await expect(page.getByTestId("work-context-summary")).toContainText("Aceitos: 2");
});

test("setup salva escolha de preset e permite customizado @presets", async ({ page }) => {
  await page.goto("/setup");

  await expect(page.getByTestId("project-name-input")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("project-preset-panel")).toBeVisible();
  await page.getByTestId("project-preset-select").selectOption("small_balloons");
  await expect(page.getByTestId("project-preset-description")).toContainText("Reduz tamanho de fonte");

  await page.getByTestId("setup-advanced-panel").locator("summary").click();
  await page.getByTestId("custom-preset-name").fill("Meu preset de teste");
  await page.getByTestId("custom-preset-create").click();
  await expect(page.getByTestId("project-preset-select")).toHaveValue(/custom_/);
  await expect(page.getByTestId("project-preset-description")).toContainText("Preset customizado");
});

test("setup mostra memoria da obra e exporta importa @work-memory", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("setup-advanced-panel").locator("summary").click();
  await expect(page.getByTestId("work-memory-panel")).toBeVisible();
  await expect(page.getByTestId("work-memory-panel")).toContainText("Termos revisados");

  await page.getByTestId("work-memory-export").click();
  await expect(page.getByTestId("work-memory-status")).toContainText("Memoria exportada");

  await page.getByTestId("work-memory-import").click();
  await expect(page.getByTestId("work-memory-status")).toContainText("Memoria importada");
});

test("processing mostra progresso percebido e metricas @performance", async ({ page }) => {
  await page.goto("/processing");

  await expect(page.getByTestId("processing-performance-panel")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("processing-perceived-steps")).toContainText("Aplicando glossario");
  await expect(page.getByTestId("processing-perceived-steps")).toContainText("Rodando QA");
  await expect(page.getByTestId("processing-pages-per-minute")).toBeVisible();
  await expect(page.getByText("100%").first().or(page.getByTestId("completion-total-time"))).toBeVisible();
  await expect(page.getByText(/1\s*\/\s*1/).or(page.getByTestId("page-status-grid"))).toBeVisible();
});

test("processing final mostra tempo total e capa da obra @performance", async ({ page }) => {
  const project = loadRealGeneratedProjectForE2E();
  project.work_context = {
    ...(project.work_context ?? {}),
    selected: true,
    work_id: project.work_context?.work_id ?? "grand-finale",
    title: project.work_context?.title ?? project.obra,
    context_loaded: project.work_context?.context_loaded ?? true,
    glossary_loaded: project.work_context?.glossary_loaded ?? false,
    glossary_entries_count: project.work_context?.glossary_entries_count ?? 0,
    risk_level: project.work_context?.risk_level ?? "medium",
    user_ignored_warning: project.work_context?.user_ignored_warning ?? false,
    cover_url: e2eCompletionCoverUrl,
  };
  await page.addInitScript((injectedProject) => {
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
  }, project);

  await page.goto("/processing");

  await expect(page.getByTestId("completion-total-time")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("completion-total-time")).not.toHaveText("0s");
  await expect(page.getByText(/Revis/i)).toHaveCount(0);
  const cover = page.getByTestId("completion-cover");
  await expect(cover).toBeVisible();
  await expect(cover).toHaveAttribute("src", e2eCompletionCoverUrl);
});

test("processing final de lote mostra capitulos e volta do preview @batch", async ({ page }) => {
  test.skip(!existsSync(realGeneratedProjectPath), `Projeto real nao encontrado: ${realGeneratedProjectPath}`);

  const project = loadRealGeneratedProjectForE2E();
  project.obra = "Projeto";
  project.capitulo = 2;
  project.mode = "auto";
  project.status = "setup";
  project.source_path = "e2e/batch-cap-2.cbz";
  project.output_path = "e2e/batch-cap-2.cbz";

  await page.addInitScript((injectedProject) => {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = injectedProject;
    (window as unknown as { __TRADUZAI_E2E_BATCH_SOURCES__?: string[] }).__TRADUZAI_E2E_BATCH_SOURCES__ = [
      "e2e/batch-cap-2.cbz",
      "e2e/batch-cap-3.cbz",
      "e2e/batch-cap-4.cbz",
    ];
  }, project);

  await page.goto("/processing");

  await expect(page.getByTestId("batch-completion-grid")).toBeVisible({ timeout: 20000 });
  await expect(page.getByTestId("batch-completion-card")).toHaveCount(3);
  await expect(page.getByTestId("batch-completion-card").filter({ hasText: "Cap. 2" })).toBeVisible();
  await expect(page.getByTestId("batch-completion-card").filter({ hasText: "Cap. 3" })).toBeVisible();
  await expect(page.getByTestId("batch-completion-card").filter({ hasText: "Cap. 4" })).toBeVisible();

  await page.getByTestId("batch-chapter-preview").nth(1).click();
  await expect(page).toHaveURL(/\/preview$/);
  await expect(page.getByText(/Capitulo 3/)).toBeVisible({ timeout: 15000 });

  await page.getByTestId("preview-return-batch").click();
  await expect(page).toHaveURL(/\/processing$/);
  await expect(page.getByTestId("batch-completion-grid")).toBeVisible();
  await expect(page.getByTestId("batch-completion-card")).toHaveCount(3);
});

test("preview permite revisar e ignorar flags de QA com motivo @phase17", async ({ page }) => {
  await page.goto("/preview");

  await expect(page.getByTestId("qa-panel")).toBeVisible();
  await expect(page.getByTestId("qa-review-report")).toBeVisible();
  await expect(page.getByTestId("qa-issue-count")).toHaveText("1");
  await expect(page.getByTestId("qa-critical-count")).toHaveText("1");
  await expect(page.getByTestId("qa-group-list")).toContainText("Ingles restante");
  await expect(page.getByTestId("qa-flag-item").first()).toContainText("Ingles restante");

  await page.getByTestId("export-panel-toggle").click();
  await expect(page.getByTestId("export-mode-options")).toContainText("Review package");
  await page.getByTestId("export-mode-review_package").click();
  await page.getByTestId("export-mode-clean").click();
  await page.getByTestId("export-button").click();
  await expect(page.getByTestId("export-block-message")).toContainText("Export limpo bloqueado");

  await page.getByTestId("qa-flag-item").first().click();
  await expect(page.getByTestId("preview-page-counter")).toHaveText("1 / 1");

  await page.getByTestId("qa-ignore-button").first().click();
  await expect(page.getByTestId("qa-save-ignore")).toBeDisabled();

  await page.getByTestId("qa-ignore-reason").fill("SFX preservado propositalmente");
  await page.getByTestId("qa-save-ignore").click();

  await expect(page.getByTestId("qa-issue-count")).toHaveText("0");
  await expect(page.getByTestId("qa-critical-count")).toHaveText("0");
  await expect(page.getByText("SFX preservado propositalmente")).toBeVisible();
});

test("preview vertical mantem zoom e controles ao rolar @phase17", async ({ page }) => {
  const injectedProject = {
    id: "preview-scroll-e2e",
    obra: "Preview Scroll E2E",
    capitulo: 1,
    paginas: Array.from({ length: 3 }, (_, index) => ({
      numero: index + 1,
      text_layers: [],
      textos: [],
    })),
    status: "done",
    source_path: "e2e/project-basic.json",
    output_path: "e2e/project-basic.json",
    totalPages: 3,
    mode: "manual",
  };

  await page.addInitScript((project) => {
    (window as unknown as { __TRADUZAI_E2E_PROJECT__?: unknown }).__TRADUZAI_E2E_PROJECT__ = project;
  }, injectedProject);

  await page.goto("/preview");

  const toolbar = page.getByTestId("preview-zoom-toolbar");
  await expect(toolbar).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("preview-page-counter")).toHaveText("1 / 3");

  await page.getByTitle("Diminuir zoom (-)").click();
  await expect(page.getByTestId("preview-zoom-value")).toHaveText("85%");

  await page.getByTestId("preview-viewport").evaluate((node) => {
    node.scrollTop = node.scrollHeight - node.clientHeight;
    node.dispatchEvent(new Event("scroll", { bubbles: true }));
  });

  await expect.poll(async () => page.getByTestId("preview-page-counter").textContent()).toBe("3 / 3");
  await expect(page.getByTestId("preview-zoom-value")).toHaveText("85%");
  await expect(toolbar).toBeVisible();
});
