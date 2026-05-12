import { useEffect, useState } from "react";
import {
  CheckCircle2,
  Download,
  Globe2,
  HardDrive,
  Languages,
  Monitor,
  Moon,
  PackageCheck,
  Sun,
  Zap,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import {
  checkModels,
  downloadModels,
  onModelsProgress,
  onModelsReady,
} from "../lib/tauri";
import {
  applyAppPreferences,
  getAppPreferences,
  saveAppPreferences,
  type AppLanguage,
  type AppThemeMode,
} from "../lib/appPreferences";

const THEME_OPTIONS: Array<{ value: AppThemeMode; label: string; icon: typeof Moon }> = [
  { value: "dark", label: "Escuro", icon: Moon },
  { value: "light", label: "Claro", icon: Sun },
  { value: "system", label: "Sistema", icon: Monitor },
];

const LANGUAGE_OPTIONS: Array<{ value: AppLanguage; label: string }> = [
  { value: "pt-BR", label: "Português" },
  { value: "en", label: "English" },
];

export function Settings() {
  const {
    gpuAvailable,
    gpuName,
    modelsReady,
    setModelsReady,
  } = useAppStore();
  const [preferences, setPreferences] = useState(() => getAppPreferences());
  const [downloading, setDownloading] = useState(false);
  const [downloadLog, setDownloadLog] = useState<string[]>([]);
  const gpuDetecting = gpuName.toLowerCase().includes("verificando");

  useEffect(() => {
    applyAppPreferences(preferences);

    let unlistenProgress: (() => void) | null = null;
    let unlistenReady: (() => void) | null = null;

    onModelsProgress((data) => {
      setDownloadLog((prev) => [...prev, data.message]);
    }).then((fn) => { unlistenProgress = fn; });

    onModelsReady((data) => {
      setDownloading(false);
      if (data.success) {
        setDownloadLog((prev) => (prev.length > 0 ? [...prev, "Pacotes prontos."] : prev));
        checkModels().then((m) => setModelsReady(m.ready));
      }
    }).then((fn) => { unlistenReady = fn; });

    return () => {
      unlistenProgress?.();
      unlistenReady?.();
    };
  }, [preferences, setModelsReady]);

  function updatePreferences(next: Partial<typeof preferences>) {
    const merged = { ...preferences, ...next };
    setPreferences(merged);
    saveAppPreferences(merged);
  }

  async function handleDownloadPackages() {
    setDownloading(true);
    setDownloadLog(["Iniciando download..."]);
    try {
      await downloadModels();
      const status = await checkModels();
      setModelsReady(status.ready);
      setDownloadLog((prev) => (prev.some((item) => item.includes("Pacotes prontos"))
        ? prev
        : [...prev, "Pacotes prontos."]));
    } catch (err) {
      setDownloadLog((prev) => [...prev, `Erro: ${err}`]);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div data-testid="settings-page" className="app-window-centered mx-auto flex min-h-full max-w-2xl flex-col justify-center p-8">
      <h2 className="mb-6 text-xl font-bold">Configurações</h2>

      <section data-testid="settings-appearance-section" className="mb-8">
        <h3 className="mb-3 text-sm font-medium text-text-secondary">Aparência</h3>
        <div className="app-card-gradient rounded-lg border border-border/80 p-4">
          <div className="mb-5 flex items-start gap-3">
            <div className="rounded-lg bg-brand/10 p-2 text-brand">
              <Globe2 size={16} />
            </div>
            <div>
              <p className="text-sm text-text-primary">Tema do app</p>
              <p className="mt-1 text-xs text-text-secondary">Escolha como a interface aparece nesta máquina.</p>
            </div>
          </div>

          <div className="mb-5 grid grid-cols-3 gap-2">
            {THEME_OPTIONS.map((option) => {
              const Icon = option.icon;
              const active = preferences.themeMode === option.value;
              return (
                <button
                  key={option.value}
                  data-testid={`settings-theme-${option.value}`}
                  onClick={() => updatePreferences({ themeMode: option.value })}
                  className={`flex items-center justify-center gap-2 rounded-lg border px-3 py-2 text-sm transition-smooth ${
                    active
                      ? "border-brand/45 bg-gradient-to-br from-brand/20 to-[#00A3FF]/10 text-brand-300"
                      : "border-border bg-bg-tertiary/80 text-text-secondary hover:border-border-strong hover:text-text-primary"
                  }`}
                >
                  <Icon size={15} />
                  {option.label}
                </button>
              );
            })}
          </div>

          <label className="block text-xs font-medium text-text-secondary" htmlFor="settings-app-language">
            Idioma do app
          </label>
          <div className="mt-2 flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 py-2">
            <Languages size={15} className="text-brand-300" />
            <select
              id="settings-app-language"
              data-testid="settings-app-language"
              value={preferences.language}
              onChange={(event) => updatePreferences({ language: event.target.value as AppLanguage })}
              className="w-full bg-transparent text-sm text-text-primary outline-none"
            >
              {LANGUAGE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <section data-testid="settings-system-section" className="mb-8">
        <h3 className="mb-3 text-sm font-medium text-text-secondary">Sistema</h3>
        <div className="app-card-gradient rounded-lg border border-border/80 px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 items-center gap-3">
              <Zap
                size={16}
                className={(gpuAvailable || gpuDetecting) ? "text-status-success" : "text-status-warning"}
              />
              <div className="min-w-0">
                <p className="text-sm">GPU</p>
                <p className="truncate text-xs text-text-secondary">{gpuName}</p>
              </div>
            </div>
            <span className={`shrink-0 rounded px-2 py-0.5 text-xs ${
              (gpuAvailable || gpuDetecting)
                ? "bg-status-success/10 text-status-success"
                : "bg-status-warning/10 text-status-warning"
            }`}>
              {gpuDetecting ? "Modo GPU" : gpuAvailable ? "CUDA ativo" : "Modo CPU"}
            </span>
          </div>
        </div>
      </section>

      <section data-testid="settings-packages-section" className="mb-8">
        <h3 className="mb-3 text-sm font-medium text-text-secondary">Pacotes</h3>
        <div className="app-card-gradient rounded-lg border border-border/80 p-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 items-center gap-3">
              <PackageCheck size={17} className={modelsReady ? "text-status-success" : "text-status-info"} />
              <div className="min-w-0">
                <p className="text-sm">Pacotes essenciais</p>
                <p className="truncate text-xs text-text-secondary">OCR, inpainting e recursos locais de imagem</p>
              </div>
            </div>
            {modelsReady ? (
              <span className="flex shrink-0 items-center gap-1.5 rounded bg-status-success/10 px-2 py-1 text-xs text-status-success">
                <CheckCircle2 size={13} />
                Pronto
              </span>
            ) : (
              <button
                data-testid="settings-download-packages"
                onClick={handleDownloadPackages}
                disabled={downloading}
                className="app-button-gradient flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-white transition-smooth disabled:cursor-not-allowed disabled:opacity-60"
              >
                {downloading ? <HardDrive size={14} /> : <Download size={14} />}
                {downloading ? "Baixando..." : "Baixar pacotes"}
              </button>
            )}
          </div>

          {downloadLog.length > 0 && (
            <pre
              data-testid="settings-package-log"
              className="mt-4 max-h-32 overflow-y-auto rounded-lg bg-bg-tertiary p-3 font-mono text-xs text-text-secondary"
            >
              {downloadLog.join("\n")}
            </pre>
          )}
        </div>
      </section>

      <section>
        <div className="space-y-1 text-center text-xs text-text-secondary/50">
          <p>TraduzAi v0.1.0 - Custo de tradução: R$0,00</p>
          <p>Imagens processadas localmente; texto traduzido via Google Translate</p>
        </div>
      </section>
    </div>
  );
}
