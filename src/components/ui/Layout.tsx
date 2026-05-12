import { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Coins,
  FlaskConical,
  Home,
  RefreshCw,
  Settings,
  Zap,
} from "lucide-react";
import { useAppStore } from "../../lib/stores/appStore";
import { openLabWindow, restartApp } from "../../lib/tauri";
import { FEATURES } from "../../lib/features";

type NavItem = {
  path: string;
  icon: typeof Home;
  label: string;
  kind?: "route" | "window";
};

export function Layout({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const workspaceMode = location.pathname.startsWith("/preview");
  const {
    credits,
    freeRemaining,
    gpuAvailable,
    gpuName,
  } = useAppStore();
  const free = freeRemaining();

  const navItems: NavItem[] = [
    { path: "/", icon: Home, label: "Início", kind: "route" },
    ...(FEATURES.lab
      ? [{ path: "/lab/home", icon: FlaskConical, label: "Lab", kind: "window" as const }]
      : []),
    { path: "/settings", icon: Settings, label: "Config", kind: "route" },
  ];

  async function handleNavClick(item: NavItem) {
    if (item.kind === "window") {
      if (!FEATURES.lab) {
        navigate("/");
        return;
      }
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

  if (workspaceMode) {
    return (
      <div
        data-testid="app-shell"
        className="app-workspace-shell flex h-screen overflow-hidden text-text-primary"
      >
        <main
          data-testid="app-main"
          className="app-main-workspace relative z-10 flex-1 overflow-hidden"
        >
          {children}
        </main>
      </div>
    );
  }

  return (
    <div
      data-testid="app-shell"
      className="app-gradient-shell flex h-screen overflow-hidden text-text-primary"
    >
      {/* Sidebar */}
      <aside
        data-testid="app-sidebar"
        className="app-sidebar-minimal relative z-10 flex w-64 flex-col"
      >
        {/* Brand */}
        <div
          className="px-5 pt-5 pb-5 cursor-pointer select-none"
          data-tauri-drag-region
          onClick={() => navigate("/")}
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-brand-300 via-brand to-[#00A3FF] flex items-center justify-center shadow-glow-brand">
              <span className="text-white font-bold text-sm tracking-tight">T</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-text-primary leading-tight tracking-tight">
                TraduzAi
              </h1>
              <p className="text-2xs text-text-muted leading-tight mt-0.5">
                Tradução automática
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          <p className="px-3 pb-2 pt-1 text-2xs text-text-muted uppercase tracking-wider font-medium">
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
                data-active={active ? "true" : "false"}
                onClick={() => void handleNavClick(item)}
                className={`group w-full flex items-center gap-3 px-3 h-9 rounded-lg text-sm
                  transition-all duration-200 ease-out-expo focus-visible:outline-none
                  ${active
                    ? "text-brand-200"
                    : "text-text-secondary hover:text-text-primary hover:bg-white/[0.03]"
                  }`}
              >
                <item.icon
                  size={16}
                  className={active ? "text-brand-300" : "text-text-muted group-hover:text-text-secondary"}
                  strokeWidth={active ? 2.25 : 1.75}
                />
                <span className={active ? "font-medium" : "font-normal"}>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status block */}
        <div className="px-3 pb-3 pt-4 space-y-3">
          <p className="px-2 text-2xs text-text-muted uppercase tracking-wider font-medium">
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
                strokeWidth={1.75}
              />
            }
            label={gpuAvailable ? gpuName : "CPU (sem GPU)"}
            valueClass="text-text-secondary truncate"
          />
        </div>

        {/* Footer */}
        <div className="px-3 pb-4 pt-2">
          <button
            onClick={() => restartApp()}
            title="Reiniciar app"
            className="w-full flex items-center justify-center gap-2 h-8 rounded-lg text-2xs
              text-text-muted hover:text-text-secondary hover:bg-white/[0.03] transition-all duration-200"
          >
            <RefreshCw size={11} strokeWidth={1.75} />
            Reiniciar app
          </button>
        </div>
      </aside>

      <main
        data-testid="app-main"
        className="app-main-unified relative z-10 flex-1 overflow-y-auto overflow-x-hidden"
      >
        {children}
      </main>
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
