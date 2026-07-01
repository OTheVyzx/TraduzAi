function isTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

export function canUseProjectDialogs() {
  return isTauriRuntime();
}

export async function openProjectDialog() {
  if (!isTauriRuntime()) {
    throw new Error("Abrir projeto pelo sistema de arquivos esta disponivel apenas no app desktop.");
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "Projeto TraduzAI", extensions: ["json"] }],
    title: "Abrir project.json",
  });
  return typeof selected === "string" ? selected : null;
}

export async function saveProjectDialog(defaultPath = "project.json") {
  if (!isTauriRuntime()) {
    throw new Error("Salvar projeto pelo sistema de arquivos esta disponivel apenas no app desktop.");
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const selected = await save({
    defaultPath,
    filters: [{ name: "Projeto TraduzAI", extensions: ["json"] }],
    title: "Salvar project.json",
  });
  return typeof selected === "string" ? selected : null;
}
