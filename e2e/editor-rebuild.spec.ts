import { expect, test } from "@playwright/test";

const criticalPatterns = [
  /getSnapshot should be cached/i,
  /Maximum update depth exceeded/i,
  /React fatal/i,
  /Zustand fatal/i,
];

test("editor Konva usa fundo limpo e layers editaveis @smoke", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/editor");

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible();
  await expect(stage).toHaveAttribute("data-base-kind", "inpaint");
  await expect(page.getByTestId("editor-professional-toolbar")).toContainText("Brush de mascara");
  await expect(page.getByTestId("editor-professional-toolbar")).toContainText("Ctrl+Z");
  await expect(page.getByTestId("editor-view-original")).toBeVisible();
  await expect(page.getByTestId("editor-tool-select")).toBeVisible();

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
    }>;
    return layers[0];
  }

  const canvas = stage.locator("canvas").first();
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) throw new Error("Canvas sem bounding box");

  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.34);
  await expect(page.getByText("Texto selecionado")).toBeVisible();

  const beforeDrag = await layerState();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.34);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.58, box.y + box.height * 0.4, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => (await layerState()).bbox[0]).not.toBe(beforeDrag.bbox[0]);

  await page.getByTitle("Texto traduzido").fill("CAMADA EDITADA");
  await expect(page.getByRole("textbox", { name: "Texto traduzido" })).toHaveValue("CAMADA EDITADA");

  await page.getByRole("button", { name: "Estilo" }).click();
  await page.getByTitle("Cor do texto").fill("#ff0000");
  await expect(page.getByTestId("editor-preview-status")).toHaveAttribute("data-status", "stale");

  await page.getByTitle("Ocultar camada de texto").click();
  await expect.poll(async () => (await layerState()).visible).toBe(false);
  await page.getByTitle("Mostrar camada de texto").click();
  await expect.poll(async () => (await layerState()).visible).toBe(true);

  await page.getByTitle("Bloquear camada de texto").click();
  await expect.poll(async () => (await layerState()).locked).toBe(true);
  const lockedBbox = (await layerState()).bbox;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.34);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.46, { steps: 8 });
  await page.mouse.up();
  expect((await layerState()).bbox).toEqual(lockedBbox);
  await page.getByTitle("Desbloquear camada de texto").click();

  await page.getByRole("button", { name: "Novo bloco" }).click();
  const beforeCreateCount = JSON.parse((await page.getByTestId("editor-stage-state").getAttribute("data-layers")) ?? "[]").length;
  await page.mouse.move(box.x + box.width * 0.18, box.y + box.height * 0.58);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.36, box.y + box.height * 0.70, { steps: 8 });
  await page.mouse.up();
  await expect
    .poll(async () => JSON.parse((await page.getByTestId("editor-stage-state").getAttribute("data-layers")) ?? "[]").length)
    .toBe(beforeCreateCount + 1);

  await page.keyboard.press("n");
  await page.mouse.move(box.x + box.width * 0.20, box.y + box.height * 0.22);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.30, box.y + box.height * 0.28, { steps: 5 });
  await page.mouse.up();
  await expect(page.getByText("Imagem: Brush")).toBeVisible();

  await page.keyboard.press("m");
  await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.22);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.28, { steps: 5 });
  await page.mouse.up();
  await expect(page.getByText("Imagem: Máscara")).toBeVisible();

  const criticalErrors = errors.filter((message) => criticalPatterns.some((pattern) => pattern.test(message)));
  expect(criticalErrors).toEqual([]);
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

test("setup avisa quando nenhuma obra foi selecionada @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("");
  await page.getByRole("button", { name: "Iniciar projeto manual" }).click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Nenhuma obra selecionada.")).toBeVisible();
  await expect(page.getByText("A traducao sera feita sem contexto.")).toBeVisible();
});

test("setup avisa quando obra esta sem contexto ativo @phase4 @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await expect(page.getByTestId("project-name-input")).toHaveValue("Fixture E2E");
  await page.getByRole("button", { name: "Iniciar projeto manual" }).click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Obra selecionada, mas glossario vazio.")).toBeVisible();

  await page.getByTestId("work-context-continue-without-context").click();
  await expect(page).toHaveURL(/\/processing$/);
});

test("setup permite adicionar e remover termo do glossario @phase5", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await page.getByRole("button", { name: "Iniciar projeto manual" }).click();
  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await page.getByRole("button", { name: "Revisar glossario" }).click();

  await page.getByPlaceholder("Termo origem").fill("Knight");
  await page.getByPlaceholder("Traducao").fill("Cavaleiro");
  await page.getByTestId("glossary-add-entry-button").click();

  await expect(page.getByText("Knight")).toBeVisible();
  await expect(page.getByText("Cavaleiro")).toBeVisible();

  await page.getByTitle("Remover termo").click();
  await expect(page.getByText("Knight")).toHaveCount(0);
});

test("setup busca contexto online e aplica candidatos @internet-context @setup", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("The Regressed Mercenary Has a Plan");
  await expect(page.getByTestId("work-context-risk")).toHaveText("alto");
  await page.getByTestId("internet-context-search").click();
  await expect(page.getByTestId("work-result-item").first()).toBeVisible();

  await page.getByTestId("work-result-item").first().click();
  await expect(page.getByTestId("internet-context-results")).toBeVisible();
  await expect(page.getByTestId("internet-context-results").getByText("anilist")).toBeVisible();
  await expect(page.getByTestId("internet-context-results").getByText("Ghislain Perdium")).toBeVisible();

  await page.getByTestId("internet-context-apply").click();
  await expect(page.getByTestId("internet-context-applied")).toContainText("2");
  await expect(page.getByTestId("glossary-editor").getByText("mana technique")).toBeVisible();
  await expect(page.getByTestId("work-context-risk")).toHaveText("medio");
});

test("setup revisa candidatos no glossario central @glossary-center", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("The Regressed Mercenary Has a Plan");
  await page.getByTestId("internet-context-search").click();
  await page.getByTestId("work-result-item").first().click();

  await expect(page.getByTestId("glossary-tab-online")).toBeVisible();
  await expect(page.getByTestId("glossary-candidate-row").filter({ hasText: "Ghislain Perdium" })).toBeVisible();

  await page
    .getByTestId("glossary-candidate-row")
    .filter({ hasText: "Ghislain Perdium" })
    .getByTestId("glossary-confirm-candidate")
    .click();

  await expect(page.getByTestId("glossary-reviewed-row").filter({ hasText: "Ghislain Perdium" })).toBeVisible();
  await expect(page.getByTestId("work-context-summary")).toContainText("1 termos");

  await page.getByTestId("glossary-tab-online").click();
  await page
    .getByTestId("glossary-candidate-row")
    .filter({ hasText: "Cavald" })
    .getByTestId("glossary-reject-candidate")
    .click();

  await expect(page.getByTestId("glossary-rejected-row").filter({ hasText: "Cavald" })).toBeVisible();
  await page.getByTestId("glossary-tab-online").click();
  await expect(page.getByTestId("glossary-online-list").getByText("Cavald")).toHaveCount(0);
});

test("setup salva escolha de preset e permite customizado @presets", async ({ page }) => {
  await page.goto("/setup");

  await expect(page.getByTestId("project-preset-panel")).toBeVisible();
  await page.getByTestId("project-preset-select").selectOption("small_balloons");
  await expect(page.getByTestId("project-preset-description")).toContainText("Reduz tamanho de fonte");

  await page.getByTestId("custom-preset-name").fill("Meu preset de teste");
  await page.getByTestId("custom-preset-create").click();
  await expect(page.getByTestId("project-preset-select")).toHaveValue(/custom_/);
  await expect(page.getByTestId("project-preset-description")).toContainText("Preset customizado");
});

test("setup mostra memoria da obra e exporta importa @work-memory", async ({ page }) => {
  await page.goto("/setup");

  await expect(page.getByTestId("work-memory-panel")).toBeVisible();
  await expect(page.getByTestId("work-memory-panel")).toContainText("Termos revisados");

  await page.getByTestId("work-memory-export").click();
  await expect(page.getByTestId("work-memory-status")).toContainText("Memoria exportada");

  await page.getByTestId("work-memory-import").click();
  await expect(page.getByTestId("work-memory-status")).toContainText("Memoria importada");
});

test("processing mostra progresso percebido e metricas @performance", async ({ page }) => {
  await page.goto("/processing");

  await expect(page.getByTestId("processing-performance-panel")).toBeVisible();
  await expect(page.getByTestId("processing-perceived-steps")).toContainText("Aplicando glossario");
  await expect(page.getByTestId("processing-perceived-steps")).toContainText("Rodando QA");
  await expect(page.getByTestId("processing-pages-per-minute")).toBeVisible();
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
