import { expect, test } from "@playwright/test";

test("text stays selected while dragging a transform handle", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/editor", { waitUntil: "domcontentloaded" });

  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible({ timeout: 120_000 });
  await page.getByText("TEXTO LIMPO").click();

  const textControls = page.getByTitle("Tamanho da fonte");
  await expect(textControls).toBeVisible();
  const floatingEditor = page.getByPlaceholder("Tradução...");
  await expect(floatingEditor).toBeVisible({ timeout: 60_000 });

  const readTextBox = async () => {
    const raw = await page.getByTestId("editor-stage-state").getAttribute("data-layers");
    const layers = JSON.parse(raw ?? "[]") as Array<{ bbox: [number, number, number, number] }>;
    return layers[0].bbox;
  };

  const before = await readTextBox();
  const canvas = stage.locator("canvas").first();
  const canvasBox = await canvas.boundingBox();
  expect(canvasBox).not.toBeNull();
  if (!canvasBox) throw new Error("Editor canvas has no bounding box");
  const canvasSize = await canvas.evaluate((node: HTMLCanvasElement) => ({ width: node.width, height: node.height }));
  const anchor = {
    x: canvasBox.x + (before[0] / canvasSize.width) * canvasBox.width,
    y: canvasBox.y + (before[1] / canvasSize.height) * canvasBox.height,
  };

  await page.mouse.move(anchor.x, anchor.y);
  await page.mouse.down();
  await expect(floatingEditor).toBeVisible({ timeout: 60_000 });
  await expect(textControls).toBeVisible();
  await page.mouse.move(anchor.x - 24, anchor.y - 20, { steps: 5 });
  await page.mouse.up();

  await expect(textControls).toBeVisible();
  await expect.poll(async () => {
    const after = await readTextBox();
    return JSON.stringify([after[2] - after[0], after[3] - after[1]]);
  }).not.toBe(JSON.stringify([before[2] - before[0], before[3] - before[1]]));
});
