import { useState, type ReactNode } from "react";
import { Bell, BookOpen, BookOpenText, Home, Settings, X } from "lucide-react";

export type StudioShellView = "home" | "library" | "reader" | "settings";

const MAIN_NAVIGATION: ReadonlyArray<{ view: StudioShellView; label: string; icon: typeof Home }> = [
  { view: "library", label: "Tradutor", icon: BookOpen },
  { view: "home", label: "Home", icon: Home },
  { view: "reader", label: "Leitor", icon: BookOpenText },
];

export function StudioAppShell({
  activeView,
  onNavigate,
  notification,
  onDismissNotification,
  children,
}: {
  activeView: StudioShellView;
  onNavigate: (view: StudioShellView) => void;
  notification?: string | null;
  onDismissNotification?: () => void;
  children: ReactNode;
}) {
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  return (
    <main className="studio-app-shell">
      <header className="studio-global-topbar">
        <div className="studio-global-brand" aria-label="TraduzAI Studio">
          <span aria-hidden="true">〽</span>
          <strong>TraduzAI Studio</strong>
        </div>
        <div className="studio-global-notifications">
          <button type="button" aria-label={notification ? "1 notificação" : "Notificações"} aria-expanded={notificationsOpen} title="Notificações" onClick={() => setNotificationsOpen((open) => !open)}>
            <Bell size={19} aria-hidden="true" />
            {notification && <span aria-hidden="true" />}
          </button>
          {notificationsOpen && <section className="studio-notification-popover" aria-label="Central de notificações">
            <header><strong>Notificações</strong><button type="button" aria-label="Fechar notificações" onClick={() => setNotificationsOpen(false)}><X size={17} /></button></header>
            {notification ? <article><p>{notification}</p><button type="button" onClick={() => { onDismissNotification?.(); setNotificationsOpen(false); }}>Dispensar</button></article> : <p>Nenhuma notificação.</p>}
          </section>}
        </div>
      </header>
      <div className="studio-app-shell-content">{children}</div>
      <footer className="studio-global-footer">
        <button
          className="studio-global-settings"
          type="button"
          aria-label="Configurações"
          title="Configurações"
          aria-current={activeView === "settings" ? "page" : undefined}
          onClick={() => onNavigate("settings")}
        >
          <Settings size={19} aria-hidden="true" />
        </button>
        <nav className="studio-global-nav" aria-label="Navegação principal">
          {MAIN_NAVIGATION.map(({ view, label, icon: Icon }) => (
            <button
              key={view}
              type="button"
              aria-current={view === activeView ? "page" : undefined}
              onClick={() => onNavigate(view)}
            >
              <Icon size={18} aria-hidden="true" />
              {label}
            </button>
          ))}
        </nav>
      </footer>
    </main>
  );
}
