export type AppThemeMode = "dark" | "light" | "system";
export type AppLanguage = "pt-BR" | "en";

const THEME_KEY = "traduzai_theme_mode";
const LANGUAGE_KEY = "traduzai_app_language";

export interface AppPreferences {
  themeMode: AppThemeMode;
  language: AppLanguage;
}

function isThemeMode(value: string | null): value is AppThemeMode {
  return value === "dark" || value === "light" || value === "system";
}

function isAppLanguage(value: string | null): value is AppLanguage {
  return value === "pt-BR" || value === "en";
}

export function getAppPreferences(): AppPreferences {
  if (typeof window === "undefined") {
    return { themeMode: "dark", language: "pt-BR" };
  }

  const themeMode = window.localStorage.getItem(THEME_KEY);
  const language = window.localStorage.getItem(LANGUAGE_KEY);
  return {
    themeMode: isThemeMode(themeMode) ? themeMode : "dark",
    language: isAppLanguage(language) ? language : "pt-BR",
  };
}

function resolveThemeMode(themeMode: AppThemeMode): "dark" | "light" {
  if (themeMode === "dark" || themeMode === "light") return themeMode;
  if (typeof window === "undefined") return "dark";
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function applyAppPreferences(preferences: AppPreferences) {
  if (typeof document === "undefined") return;
  const resolvedTheme = resolveThemeMode(preferences.themeMode);
  document.documentElement.dataset.themeMode = preferences.themeMode;
  document.documentElement.dataset.theme = resolvedTheme;
  document.documentElement.lang = preferences.language;
  document.documentElement.style.colorScheme = resolvedTheme;
}

export function saveAppPreferences(preferences: AppPreferences) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(THEME_KEY, preferences.themeMode);
    window.localStorage.setItem(LANGUAGE_KEY, preferences.language);
  }
  applyAppPreferences(preferences);
}

export function watchSystemTheme(preferences: AppPreferences): () => void {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia("(prefers-color-scheme: light)");
  const handleChange = () => applyAppPreferences(preferences);
  query.addEventListener?.("change", handleChange);
  return () => query.removeEventListener?.("change", handleChange);
}
