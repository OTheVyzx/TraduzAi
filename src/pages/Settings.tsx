import { useState, useEffect } from "react";
import {
  Cpu, Zap, HardDrive, Globe, Save, CheckCircle2,
  RefreshCw, AlertTriangle, Bot, Download, ExternalLink,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import {
  saveSettings, loadSettings, checkOllama, createTranslatorModel,
  downloadModels, onModelsProgress, onModelsReady, checkModels,
} from "../lib/tauri";

export function Settings() {
  const {
    gpuAvailable, gpuName, modelsReady, credits,
    ollamaRunning, ollamaModels, ollamaHasTranslator,
    setOllamaStatus, setModelsReady,
  } = useAppStore();

  const [ollamaModel, setOllamaModel] = useState("traduzai-translator");
  const [ollamaHost, setOllamaHost] = useState("http://localhost:11434");
  const [defaultLang, setDefaultLang] = useState("pt-BR");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [creatingModel, setCreatingModel] = useState(false);
  const [createLog, setCreateLog] = useState("");
  const [confirmCreate, setConfirmCreate] = useState(false);
  const [checkingOllama, setCheckingOllama] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadLog, setDownloadLog] = useState<string[]>([]);
  const gpuDetecting = gpuName.toLowerCase().includes("verificando");

  useEffect(() => {
    loadSettings().then((s) => {
      setOllamaModel(s.ollama_model || "traduzai-translator");
      setOllamaHost(s.ollama_host || "http://localhost:11434");
      setDefaultLang(s.idioma_destino || "pt-BR");
    });

    let unlistenProgress: (() => void) | null = null;
    let unlistenReady: (() => void) | null = null;

    onModelsProgress((data) => {
      setDownloadLog((prev) => [...prev, data.message]);
    }).then((fn) => { unlistenProgress = fn; });

    onModelsReady((data) => {
      setDownloading(false);
      if (data.success) {
        checkModels().then((m) => setModelsReady(m.ready));
      }
    }).then((fn) => { unlistenReady = fn; });

    return () => {
      unlistenProgress?.();
      unlistenReady?.();
    };
  }, []);

  async function handleDownloadModels() {
    setDownloading(true);
    setDownloadLog(["Iniciando download..."]);
    try {
      await downloadModels();
    } catch (err: any) {
      setDownloadLog((prev) => [...prev, `Erro: ${err}`]);
      setDownloading(false);
    }
  }

  async function handleRefreshOllama() {
    setCheckingOllama(true);
    try {
      const status = await checkOllama();
      setOllamaStatus(status.running, status.models, status.has_translator);
    } finally {
      setCheckingOllama(false);
    }
  }

  async function handleCreateModel() {
    if (!confirmCreate) {
      setConfirmCreate(true);
      return;
    }
    setConfirmCreate(false);
    setCreatingModel(true);
    setCreateLog("");
    try {
      const msg = await createTranslatorModel();
      setCreateLog(msg);
    } catch (err: any) {
      setCreateLog(`Erro: ${err}`);
    } finally {
      setCreatingModel(false);
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      await saveSettings({
        ollama_model: ollamaModel,
        ollama_host: ollamaHost,
        idioma_destino: defaultLang,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold mb-6">Configurações</h2>

      {/* LLM Local — Ollama */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-text-secondary mb-3">Tradução Local (Ollama)</h3>
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-4 space-y-4">

          {/* Status */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bot
                size={18}
                className={ollamaRunning ? "text-status-success" : "text-status-error"}
              />
              <div>
                <p className="text-sm font-medium">
                  {ollamaRunning ? "Ollama rodando" : "Ollama offline"}
                </p>
                <p className="text-xs text-text-secondary">
                  {ollamaRunning
                    ? `${ollamaModels.length} modelo(s) disponível(is)`
                    : "Instale o Ollama para tradução gratuita"}
                </p>
              </div>
            </div>
            <button
              onClick={handleRefreshOllama}
              disabled={checkingOllama}
              className="p-2 text-text-secondary hover:text-text-primary transition-smooth"
              title="Verificar Ollama"
            >
              <RefreshCw size={16} className={checkingOllama ? "animate-spin" : ""} />
            </button>
          </div>

          {/* Not installed */}
          {!ollamaRunning && (
            <div className="flex items-start gap-3 p-3 bg-status-warning/5 border border-status-warning/20 rounded-lg">
              <AlertTriangle size={16} className="text-status-warning mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm text-status-warning font-medium">Ollama não detectado</p>
                <p className="text-xs text-text-secondary mt-1">
                  Ollama é necessário para tradução local gratuita.
                  Baixe, instale e inicie o Ollama, depois clique em atualizar.
                </p>
                <a
                  href="https://ollama.com/download"
                  className="inline-flex items-center gap-1 text-xs text-accent-cyan mt-2 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink size={12} />
                  ollama.com/download
                </a>
              </div>
            </div>
          )}

          {/* Model setup */}
          {ollamaRunning && (
            <>
              {/* traduzai-translator status */}
              <div className="flex items-center justify-between p-3 bg-bg-tertiary rounded-lg">
                <div className="flex items-center gap-2">
                  {ollamaHasTranslator ? (
                    <CheckCircle2 size={16} className="text-status-success" />
                  ) : (
                    <AlertTriangle size={16} className="text-status-warning" />
                  )}
                  <div>
                    <p className="text-sm font-medium">traduzai-translator</p>
                    <p className="text-xs text-text-secondary">
                      {ollamaHasTranslator
                        ? "Modelo especializado instalado"
                        : "Não instalado — recomendado para melhor qualidade"}
                    </p>
                  </div>
                </div>
                {!ollamaHasTranslator && (
                  <button
                    onClick={handleCreateModel}
                    disabled={creatingModel}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-smooth disabled:opacity-50
                      ${confirmCreate
                        ? "bg-status-warning/20 text-status-warning hover:bg-status-warning/30"
                        : "bg-accent-purple/10 text-accent-purple hover:bg-accent-purple/20"}`}
                  >
                    <Download size={13} />
                    {creatingModel ? "Abrindo terminal..." : confirmCreate ? "Confirmar e abrir terminal" : "Criar modelo"}
                  </button>
                )}
              </div>

              {/* Confirmation preview */}
              {confirmCreate && !ollamaHasTranslator && (
                <div className="p-3 bg-status-warning/5 border border-status-warning/20 rounded-lg space-y-2">
                  <p className="text-xs text-status-warning font-medium">Isso abrirá um terminal PowerShell com:</p>
                  <pre className="text-xs text-text-secondary bg-bg-primary rounded p-2 font-mono">
{`ollama pull qwen2.5:3b
ollama create traduzai-translator -f ...Modelfile`}
                  </pre>
                  <p className="text-xs text-text-secondary">
                    Download ~1,9 GB. Após fechar o terminal, clique em <strong>Atualizar</strong> para detectar o modelo.
                  </p>
                  <button
                    onClick={() => setConfirmCreate(false)}
                    className="text-xs text-text-secondary hover:text-text-primary transition-smooth"
                  >
                    Cancelar
                  </button>
                </div>
              )}

              {createLog && (
                <pre className="text-xs text-text-secondary bg-bg-tertiary rounded p-3 whitespace-pre-wrap font-mono">
                  {createLog}
                </pre>
              )}

              {/* Model selector */}
              <div>
                <label className="text-xs text-text-secondary block mb-1.5">
                  Modelo ativo
                </label>
                <select
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-tertiary border border-white/5 rounded-lg text-sm
                    text-text-primary focus:outline-none focus:border-accent-purple/30 transition-smooth"
                >
                  {ollamaHasTranslator && (
                    <option value="traduzai-translator">
                      traduzai-translator ⭐ (recomendado)
                    </option>
                  )}
                  {ollamaModels
                    .filter((m) => !m.includes("traduzai-translator"))
                    .map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                </select>
              </div>

              {/* Ollama host */}
              <div>
                <label className="text-xs text-text-secondary block mb-1.5">
                  Endereço Ollama
                </label>
                <input
                  type="text"
                  value={ollamaHost}
                  onChange={(e) => setOllamaHost(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-tertiary border border-white/5 rounded-lg text-sm
                    text-text-primary font-mono focus:outline-none focus:border-accent-purple/30 transition-smooth"
                />
              </div>
            </>
          )}
        </div>
      </section>

      {/* System status */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-text-secondary mb-3">Sistema</h3>
        <div className="bg-bg-secondary border border-white/5 rounded-xl divide-y divide-white/5">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3">
              <Zap size={16} className={(gpuAvailable || gpuDetecting) ? "text-status-success" : "text-status-warning"} />
              <div>
                <p className="text-sm">GPU</p>
                <p className="text-xs text-text-secondary">{gpuName}</p>
              </div>
            </div>
            <span className={`text-xs px-2 py-0.5 rounded ${(gpuAvailable || gpuDetecting) ? "bg-status-success/10 text-status-success" : "bg-status-warning/10 text-status-warning"}`}>
              {gpuDetecting ? "Modo GPU" : gpuAvailable ? "CUDA ativo" : "Modo CPU"}
            </span>
          </div>

          <div>
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-3">
                <HardDrive size={16} className={modelsReady ? "text-status-success" : "text-status-info"} />
                <div>
                  <p className="text-sm">Modelos OCR</p>
                  <p className="text-xs text-text-secondary">EasyOCR + inpainting local (primeira execução pode baixar modelos)</p>
                </div>
              </div>
              {modelsReady ? (
                <span className="text-xs px-2 py-0.5 rounded bg-status-success/10 text-status-success">Pronto</span>
              ) : (
                <button
                  onClick={handleDownloadModels}
                  disabled={downloading}
                  className="text-xs px-3 py-1 rounded bg-accent-purple/10 text-accent-purple hover:bg-accent-purple/20 transition-smooth disabled:opacity-50"
                >
                  {downloading ? "Baixando..." : "Baixar"}
                </button>
              )}
            </div>
            {downloadLog.length > 0 && (
              <div className="px-4 pb-3">
                <pre className="text-xs text-text-secondary bg-bg-tertiary rounded p-3 whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                  {downloadLog.join("\n")}
                </pre>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3">
              <Cpu size={16} className="text-status-info" />
              <div>
                <p className="text-sm">Créditos</p>
                <p className="text-xs text-text-secondary">{credits} disponíveis</p>
              </div>
            </div>
            <button className="text-xs px-3 py-1 rounded bg-accent-purple/10 text-accent-purple hover:bg-accent-purple/20 transition-smooth">
              Comprar
            </button>
          </div>
        </div>
      </section>

      {/* Language + Save */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-text-secondary mb-3">Idioma padrão</h3>
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <Globe size={14} className="text-text-secondary" />
            <p className="text-sm">Idioma de destino</p>
          </div>
          <select
            value={defaultLang}
            onChange={(e) => setDefaultLang(e.target.value)}
            className="w-full px-3 py-2 bg-bg-tertiary border border-white/5 rounded-lg text-sm
              text-text-primary focus:outline-none focus:border-accent-purple/30 transition-smooth mb-4"
          >
            <option value="pt-BR">Português (Brasil)</option>
            <option value="es">Español</option>
            <option value="en">English</option>
          </select>

          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-accent-purple hover:bg-accent-purple-dark
              text-white text-sm rounded-lg transition-smooth disabled:opacity-50"
          >
            {saved ? <><CheckCircle2 size={14} /> Salvo!</> : <><Save size={14} /> {saving ? "Salvando..." : "Salvar"}</>}
          </button>
        </div>
      </section>

      <section>
        <div className="text-center text-xs text-text-secondary/50 space-y-1">
          <p>TraduzAi v0.1.0 — Custo de tradução: R$0,00</p>
          <p>100% local — nenhum arquivo ou texto enviado a servidores</p>
        </div>
      </section>
    </div>
  );
}
