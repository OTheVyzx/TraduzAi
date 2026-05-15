import {
  sanitizeTextStylePresets,
  type EditorTextStylePreset,
} from "./editorTextStylePresets";
import { isE2E, loadSettings, saveSettings } from "./tauri";

const STORAGE_KEY = "traduzai_editor_text_style_presets";

function readLocalPresets() {
  if (typeof localStorage === "undefined") return [];
  try {
    return sanitizeTextStylePresets(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"), "custom");
  } catch {
    return [];
  }
}

function writeLocalPresets(presets: EditorTextStylePreset[]) {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeTextStylePresets(presets, "custom")));
}

export async function loadCustomTextStylePresets(): Promise<EditorTextStylePreset[]> {
  const localPresets = readLocalPresets();
  if (isE2E()) return localPresets;

  try {
    const settings = await loadSettings();
    const settingsPresets = sanitizeTextStylePresets(settings.editor_text_presets, "custom");
    if (settingsPresets.length > 0) {
      writeLocalPresets(settingsPresets);
      return settingsPresets;
    }
  } catch {
    return localPresets;
  }

  return localPresets;
}

export async function saveCustomTextStylePresets(presets: EditorTextStylePreset[]): Promise<void> {
  const sanitized = sanitizeTextStylePresets(presets, "custom").filter((preset) => preset.kind === "custom");
  writeLocalPresets(sanitized);
  if (isE2E()) return;

  try {
    const settings = await loadSettings();
    await saveSettings({
      ...settings,
      idioma_origem: settings.idioma_origem ?? "en",
      idioma_destino: settings.idioma_destino ?? "pt-BR",
      editor_text_presets: sanitized,
    });
  } catch (error) {
    console.warn("[text-presets] falha ao salvar presets customizados em settings:", error);
  }
}
