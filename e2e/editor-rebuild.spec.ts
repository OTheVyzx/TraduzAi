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

test("setup avisa quando obra esta sem contexto ativo @phase4", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await expect(page.getByTestId("project-name-input")).toHaveValue("Fixture E2E");
  await page.getByRole("button", { name: "Iniciar projeto manual" }).click();

  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await expect(page.getByText("Esta obra esta sem glossario ativo.")).toBeVisible();

  await page.getByTestId("work-context-continue-without-context").click();
  await expect(page).toHaveURL(/\/processing$/);
});

test("setup permite adicionar e remover termo do glossario @phase5", async ({ page }) => {
  await page.goto("/setup");

  await page.getByTestId("project-name-input").fill("Fixture E2E");
  await page.getByRole("button", { name: "Iniciar projeto manual" }).click();
  await expect(page.getByTestId("work-context-warning-modal")).toBeVisible();
  await page.getByRole("button", { name: "Editar manualmente" }).click();

  await page.getByPlaceholder("Termo origem").fill("Knight");
  await page.getByPlaceholder("Traducao").fill("Cavaleiro");
  await page.getByTestId("glossary-add-entry-button").click();

  await expect(page.getByText("Knight")).toBeVisible();
  await expect(page.getByText("Cavaleiro")).toBeVisible();

  await page.getByTitle("Remover termo").click();
  await expect(page.getByText("Knight")).toHaveCount(0);
});

test("preview permite revisar e ignorar flags de QA com motivo @phase17", async ({ page }) => {
  await page.goto("/preview");

  await expect(page.getByTestId("qa-panel")).toBeVisible();
  await expect(page.getByTestId("qa-issue-count")).toHaveText("1");
  await expect(page.getByText("Ingles restante")).toBeVisible();

  await page.getByTestId("qa-flag-item").first().click();
  await expect(page.getByTestId("preview-page-counter")).toHaveText("1 / 1");

  await page.getByTestId("qa-ignore-button").first().click();
  await expect(page.getByTestId("qa-save-ignore")).toBeDisabled();

  await page.getByTestId("qa-ignore-reason").fill("SFX preservado propositalmente");
  await page.getByTestId("qa-save-ignore").click();

  await expect(page.getByTestId("qa-issue-count")).toHaveText("0");
  await expect(page.getByText("SFX preservado propositalmente")).toBeVisible();
});
