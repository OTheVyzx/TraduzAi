import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Home } from "./pages/Home";
import { Setup } from "./pages/Setup";
import { Processing } from "./pages/Processing";
import { Preview } from "./pages/Preview";
import { Settings } from "./pages/Settings";
import { Editor } from "./pages/Editor";
import { Lab } from "./pages/Lab";
import { Layout } from "./components/ui/Layout";
import { BootSplash } from "./components/ui/BootSplash";
import { useAppStore } from "./lib/stores/appStore";
import { checkModels, getCredits, checkOllama, getSystemProfile, onPipelineProgress } from "./lib/tauri";
import { installE2EFixtureProject } from "./lib/e2e/fixtureProject";
import { FEATURES } from "./lib/features";

const LAB_WINDOW_MODE_KEY = "traduzai-window-mode";

function AppRoutes() {
  const location = useLocation();
  const queryMode = new URLSearchParams(location.search).get("window");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (!FEATURES.lab) {
      window.sessionStorage.removeItem(LAB_WINDOW_MODE_KEY);
      return;
    }

    if (queryMode === "lab") {
      window.sessionStorage.setItem(LAB_WINDOW_MODE_KEY, "lab");
    }
  }, [queryMode]);

  const standaloneLab = FEATURES.lab && (
    queryMode === "lab"
    || (typeof window !== "undefined"
      && window.sessionStorage.getItem(LAB_WINDOW_MODE_KEY) === "lab")
  );

  if (standaloneLab) {
    return (
      <Routes>
        <Route
          path="/lab/*"
          element={
            <div className="h-screen overflow-y-auto overflow-x-hidden bg-bg-primary bg-noise">
              <Lab />
            </div>
          }
        />
        <Route
          path="*"
          element={
            <Navigate
              to={{ pathname: "/lab/home", search: "?window=lab" }}
              replace
            />
          }
        />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route path="/editor" element={<Editor />} />
      <Route
        path="/*"
        element={
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/processing" element={<Processing />} />
              <Route path="/preview" element={<Preview />} />
              {FEATURES.lab ? (
                <Route path="/lab/*" element={<Lab />} />
              ) : (
                <Route path="/lab/*" element={<Navigate to="/" replace />} />
              )}
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
        }
      />
    </Routes>
  );
}

export default function App() {
  const e2eMode = ((import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_E2E ?? "") === "1";
  const setSystemProfile = useAppStore((s) => s.setSystemProfile);
  const setModelsReady = useAppStore((s) => s.setModelsReady);
  const setCredits = useAppStore((s) => s.setCredits);
  const setOllamaStatus = useAppStore((s) => s.setOllamaStatus);
  const [bootState, setBootState] = useState({
    ready: e2eMode,
    progress: e2eMode ? 1 : 0.08,
    message: e2eMode ? "Pronto" : "Preparando ambiente...",
  });

  const setPipeline = useAppStore((s) => s.setPipeline);
  const appendPipelineLog = useAppStore((s) => s.appendPipelineLog);

  useEffect(() => {
    if (e2eMode) {
      installE2EFixtureProject();
      return;
    }
    let unlisten: (() => void) | undefined;
    let lastLoggedStep: string | null = null;

    async function setup() {
      unlisten = (await onPipelineProgress((progress) => {
        setPipeline(progress);
        appendPipelineLog({
          level: progress.step !== lastLoggedStep ? "step" : "progress",
          step: progress.step as any,
          current_page: progress.current_page,
          total_pages: progress.total_pages,
          overall_progress: progress.overall_progress,
          step_progress: progress.step_progress,
          message: progress.message,
        });
        lastLoggedStep = progress.step;
      })) as unknown as () => void;
    }

    setup();
    return () => unlisten?.();
  }, [appendPipelineLog, e2eMode, setPipeline]);

  useEffect(() => {
    if (e2eMode) return;
    let cancelled = false;
    let revealTimer: number | null = null;
    let completedSteps = 0;
    const totalSteps = 4;

    const markStep = (message: string) => {
      completedSteps += 1;
      if (cancelled) return;
      setBootState({
        ready: false,
        progress: Math.min(0.94, 0.12 + (completedSteps / totalSteps) * 0.78),
        message,
      });
    };

    async function init() {
      setBootState({
        ready: false,
        progress: 0.12,
        message: "Carregando servicos locais...",
      });

      try {
        const tasks = [
          getSystemProfile()
            .then((systemProfile) => {
              if (cancelled) return;
              setSystemProfile(systemProfile);
              markStep("Hardware carregado");
            })
            .catch((error) => {
              console.error("[TraduzAi] System profile init error:", error);
              markStep("Hardware indisponivel");
            }),
          checkModels()
            .then((models) => {
              if (cancelled) return;
              setModelsReady(models.ready);
              markStep(models.ready ? "Modelos locais prontos" : "Modelos locais verificados");
            })
            .catch((error) => {
              console.error("[TraduzAi] Models init error:", error);
              markStep("Modelos locais verificados");
            }),
          getCredits()
            .then((credits) => {
              if (cancelled) return;
              setCredits(credits.credits, credits.weekly_used);
              markStep("Creditos sincronizados");
            })
            .catch((error) => {
              console.error("[TraduzAi] Credits init error:", error);
              markStep("Creditos indisponiveis");
            }),
          checkOllama()
            .then((ollama) => {
              if (cancelled) return;
              setOllamaStatus(ollama.running, ollama.models, ollama.has_translator);
              markStep(ollama.running ? "Ollama verificado" : "Ollama offline");
            })
            .catch((error) => {
              console.error("[TraduzAi] Ollama init error:", error);
              markStep("Ollama indisponivel");
            }),
        ];

        await Promise.allSettled(tasks);

        if (cancelled) return;
        setBootState({
          ready: false,
          progress: 1,
          message: "Entrando no app...",
        });
        revealTimer = window.setTimeout(() => {
          if (cancelled) return;
          setBootState({
            ready: true,
            progress: 1,
            message: "Pronto",
          });
        }, 180);
      } catch (err) {
        console.error("[TraduzAi] Init error:", err);
        if (!cancelled) {
          setBootState({
            ready: false,
            progress: 1,
            message: "Entrando no app...",
          });
          revealTimer = window.setTimeout(() => {
            if (cancelled) return;
            setBootState({
              ready: true,
              progress: 1,
              message: "Pronto",
            });
          }, 180);
        }
      }
    }

    init();
    return () => {
      cancelled = true;
      if (revealTimer !== null) {
        window.clearTimeout(revealTimer);
      }
    };
  }, [e2eMode, setCredits, setModelsReady, setOllamaStatus, setSystemProfile]);

  if (!bootState.ready) {
    return <BootSplash progress={bootState.progress} message={bootState.message} />;
  }

  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
