import { lazy, Suspense, useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Home } from "./pages/Home";
import { Layout } from "./components/ui/Layout";
import { useAppStore } from "./lib/stores/appStore";
import { checkModels, getCredits, getSystemProfile, onPipelineProgress, warmupVisualStack } from "./lib/tauri";
import { installE2EFixtureProject } from "./lib/e2e/fixtureProject";
import { FEATURES } from "./lib/features";
import { applyAppPreferences, getAppPreferences, watchSystemTheme } from "./lib/appPreferences";

const LAB_WINDOW_MODE_KEY = "traduzai-window-mode";
const Setup = lazy(() => import("./pages/Setup").then((module) => ({ default: module.Setup })));
const Processing = lazy(() => import("./pages/Processing").then((module) => ({ default: module.Processing })));
const Preview = lazy(() => import("./pages/Preview").then((module) => ({ default: module.Preview })));
const Settings = lazy(() => import("./pages/Settings").then((module) => ({ default: module.Settings })));
const Editor = lazy(() => import("./pages/Editor").then((module) => ({ default: module.Editor })));
const Lab = lazy(() => import("./pages/Lab").then((module) => ({ default: module.Lab })));

function RouteFallback() {
  return (
    <div
      data-testid="route-pending"
      className="flex min-h-full items-center justify-center px-6 py-10 text-center text-sm text-text-secondary"
    >
      Preparando tela...
    </div>
  );
}

function isTauriRuntime() {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)
  );
}

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
      <div className="h-screen overflow-y-auto overflow-x-hidden bg-bg-primary bg-noise">
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route
              path="/lab/*"
              element={
                <Lab />
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
        </Suspense>
      </div>
    );
  }

  return (
    <Routes>
      <Route
        path="/editor"
        element={
          <Suspense fallback={<RouteFallback />}>
            <Editor />
          </Suspense>
        }
      />
      <Route
        path="/*"
        element={
          <Layout>
            <Suspense fallback={<RouteFallback />}>
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
            </Suspense>
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

  const setPipeline = useAppStore((s) => s.setPipeline);
  const appendPipelineLog = useAppStore((s) => s.appendPipelineLog);

  useEffect(() => {
    const preferences = getAppPreferences();
    applyAppPreferences(preferences);
    return watchSystemTheme(preferences);
  }, []);

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
    if (!isTauriRuntime()) return;
    let cancelled = false;
    const initTimer = window.setTimeout(() => {
      const tasks = [
        getSystemProfile()
          .then((systemProfile) => {
            if (!cancelled) setSystemProfile(systemProfile);
          })
          .catch((error) => {
            console.error("[TraduzAi] System profile init error:", error);
          }),
        checkModels()
          .then((models) => {
            if (!cancelled) setModelsReady(models.ready);
          })
          .catch((error) => {
            console.error("[TraduzAi] Models init error:", error);
          }),
        getCredits()
          .then((credits) => {
            if (!cancelled) setCredits(credits.credits, credits.weekly_used);
          })
          .catch((error) => {
            console.error("[TraduzAi] Credits init error:", error);
          }),
      ];

      void Promise.allSettled(tasks);
    }, 250);
    const warmupTimer = window.setTimeout(() => {
      if (cancelled) return;
      warmupVisualStack().catch((error) => {
        console.error("[TraduzAi] Warmup visual em segundo plano falhou:", error);
      });
    }, 2000);

    return () => {
      cancelled = true;
      window.clearTimeout(initTimer);
      window.clearTimeout(warmupTimer);
    };
  }, [e2eMode, setCredits, setModelsReady, setSystemProfile]);

  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
