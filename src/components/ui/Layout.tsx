import { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Bot, Coins, FlaskConical, Home, RefreshCw, Settings, Zap } from "lucide-react";
import { useAppStore } from "../../lib/stores/appStore";
import { openLabWindow, restartApp } from "../../lib/tauri";

type NavItem = {
  path: string;
  icon: typeof Home;
  label: string;
  kind?: "route" | "window";
};

export function Layout({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { credits, freeRemaining, gpuAvailable, gpuName, ollamaRunning, ollamaHasTranslator } = useAppStore();
  const free = freeRemaining();

  const navItems: NavItem[] = [
    { path: "/", icon: Home, label: "Inicio", kind: "route" },
    { path: "/lab/home", icon: FlaskConical, label: "Lab", kind: "window" },
    { path: "/settings", icon: Settings, label: "Config", kind: "route" },
  ];

  async function handleNavClick(item: NavItem) {
    if (item.kind === "window") {
      try {
        await openLabWindow();
      } catch (error) {
        console.error("[TraduzAi] Falha ao abrir janela do Lab:", error);
        navigate(item.path);
      }
      return;
    }

    navigate(item.path);
  }

  return (
    <div className="flex h-screen bg-bg-primary">
      <aside className="w-56 bg-bg-secondary border-r border-white/5 flex flex-col">
        <div
          className="px-4 py-4 border-b border-white/5 cursor-pointer select-none"
          data-tauri-drag-region
          onClick={() => navigate("/")}
        >
          <h1 className="text-lg font-bold text-accent-purple">TraduzAi</h1>
          <p className="text-xs text-text-secondary mt-0.5">Traducao automatica</p>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-1">
          {navItems.map((item) => {
            const active =
              item.path === "/"
                ? location.pathname === item.path
                : location.pathname === item.path
                  || location.pathname.startsWith(`${item.path}/`)
                  || (item.path === "/lab/home" && location.pathname.startsWith("/lab"));

            return (
              <button
                key={item.path}
                onClick={() => void handleNavClick(item)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-smooth
                  ${active
                    ? "bg-accent-purple/10 text-accent-purple"
                    : "text-text-secondary hover:text-text-primary hover:bg-bg-tertiary"
                  }`}
              >
                <item.icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="px-3 pb-2">
          <button
            onClick={() => restartApp()}
            title="Reiniciar app"
            className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs
              text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-smooth"
          >
            <RefreshCw size={13} />
            Reiniciar app
          </button>
        </div>

        <div className="px-3 py-3 border-t border-white/5 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Coins size={14} className="text-accent-purple" />
            <span className="text-text-secondary">
              {credits > 0 ? `${credits} creditos` : `${free} pg gratis`}
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <Zap
              size={14}
              className={gpuAvailable ? "text-status-success" : "text-status-warning"}
            />
            <span className="text-text-secondary truncate">
              {gpuAvailable ? gpuName : "CPU (sem GPU)"}
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <Bot
              size={14}
              className={
                ollamaHasTranslator
                  ? "text-status-success"
                  : ollamaRunning
                    ? "text-status-warning"
                    : "text-status-error"
              }
            />
            <span
              className={
                ollamaHasTranslator
                  ? "text-status-success"
                  : ollamaRunning
                    ? "text-status-warning"
                    : "text-status-error"
              }
            >
              {ollamaHasTranslator
                ? "LLM pronto"
                : ollamaRunning
                  ? "Sem modelo"
                  : "Ollama offline"}
            </span>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}
