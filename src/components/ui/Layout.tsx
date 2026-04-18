import { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Bot,
  Coins,
  FlaskConical,
  Home,
  RefreshCw,
  Settings,
  Zap,
} from "lucide-react";
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
  const {
    credits,
    freeRemaining,
    gpuAvailable,
    gpuName,
    ollamaRunning,
    ollamaHasTranslator,
  } = useAppStore();
  const free = freeRemaining();

  const navItems: NavItem[] = [
    { path: "/", icon: Home, label: "Início", kind: "route" },
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

  const ollamaStatus = ollamaHasTranslator
    ? { label: "LLM pronto", color: "text-status-success", dot: "bg-status-success" }
    : ollamaRunning
      ? { label: "Sem modelo", color: "text-status-warning", dot: "bg-status-warning" }
      : { label: "Ollama offline", color: "text-status-error", dot: "bg-status-error" };

  return (
    <div className="flex h-screen bg-bg-primary">
      <aside className="w-60 bg-bg-base border-r border-border flex flex-col">
        {/* Brand */}
        <div
          className="px-5 pt-5 pb-4 cursor-pointer select-none"
          data-tauri-drag-region
          onClick={() => navigate("/")}
        >
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-gradient-to-br from-brand to-brand-700 flex items-center justify-center shadow-glow-brand">
              <span className="text-white font-bold text-sm">T</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-text-primary leading-tight">
                TraduzAi
              </h1>
              <p className="text-2xs text-text-muted leading-tight mt-0.5">
                Tradução automática
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-2 space-y-0.5">
          <p className="px-3 pb-2 pt-2 text-2xs text-text-muted uppercase tracking-wider font-medium">
            Navegação
          </p>
          {navItems.map((item) => {
            const active =
              item.path === "/"
                ? location.pathname === item.path
                : location.pathname === item.path ||
                  location.pathname.startsWith(`${item.path}/`) ||
                  (item.path === "/lab/home" && location.pathname.startsWith("/lab"));

            return (
              <button
                key={item.path}
                onClick={() => void handleNavClick(item)}
                className={`group w-full flex items-center gap-3 px-3 h-9 rounded-md text-sm
                  transition-all duration-180 ease-out-expo focus-visible:outline-none
                  ${active
                    ? "bg-brand/12 text-brand-200 shadow-[inset_2px_0_0_0_theme(colors.brand.DEFAULT)]"
                    : "text-text-secondary hover:text-text-primary hover:bg-white/[0.04]"
                  }`}
              >
                <item.icon
                  size={16}
                  className={active ? "text-brand-300" : ""}
                  strokeWidth={active ? 2.25 : 2}
                />
                <span className="font-medium">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status block */}
        <div className="px-3 pb-3 pt-4 border-t border-border space-y-2.5">
          <p className="px-2 text-2xs text-text-muted uppercase tracking-wider font-medium mb-2">
            Status
          </p>

          <StatusRow
            icon={<Coins size={13} className="text-brand-300" />}
            label={credits > 0 ? `${credits} créditos` : `${free} pg grátis`}
            valueClass="text-text-primary"
          />
          <StatusRow
            icon={
              <Zap
                size={13}
                className={gpuAvailable ? "text-status-success" : "text-status-warning"}
              />
            }
            label={gpuAvailable ? gpuName : "CPU (sem GPU)"}
            valueClass="text-text-secondary truncate"
          />
          <StatusRow
            icon={
              <span className="relative flex w-2 h-2">
                <span
                  className={`absolute inset-0 rounded-full ${ollamaStatus.dot} ${
                    ollamaHasTranslator ? "animate-pulse-glow" : ""
                  }`}
                />
              </span>
            }
            label={
              <span className="flex items-center gap-1.5">
                <Bot size={12} className={ollamaStatus.color} />
                <span className={ollamaStatus.color}>{ollamaStatus.label}</span>
              </span>
            }
            valueClass=""
          />
        </div>

        {/* Footer */}
        <div className="px-3 pb-4 pt-2 border-t border-border">
          <button
            onClick={() => restartApp()}
            title="Reiniciar app"
            className="w-full flex items-center justify-center gap-2 h-8 rounded-md text-2xs
              text-text-muted hover:text-text-secondary hover:bg-white/[0.04] transition-all duration-180"
          >
            <RefreshCw size={11} />
            Reiniciar app
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto overflow-x-hidden">{children}</main>
    </div>
  );
}

function StatusRow({
  icon,
  label,
  valueClass,
}: {
  icon: ReactNode;
  label: ReactNode;
  valueClass: string;
}) {
  return (
    <div className="flex items-center gap-2.5 px-2 text-xs">
      <span className="shrink-0 flex items-center justify-center w-4">{icon}</span>
      <span className={`flex-1 min-w-0 ${valueClass}`}>{label}</span>
    </div>
  );
}
