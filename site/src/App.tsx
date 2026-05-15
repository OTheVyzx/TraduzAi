import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { BrowserRouter, Link, Navigate, NavLink, Route, Routes, useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  Archive,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Bold,
  Brush,
  ChevronLeft,
  CheckCircle2,
  Clock,
  Coins,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  Gauge,
  Image,
  Italic,
  LayoutDashboard,
  LinkIcon,
  Languages,
  Layers,
  LogOut,
  MousePointer2,
  PauseCircle,
  Plus,
  RotateCcw,
  Scissors,
  Server,
  Settings,
  Sparkles,
  Search,
  Star,
  Trash2,
  Undo2,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { emptyProjectConfig, type GlossaryCandidate, type WebProjectConfig, type WebProjectMode } from "./projectConfig";
import { setupApi, type WorkSearchResult } from "./projectSetupApi";
import { assetUrl, projectApi, type ProjectLayerMap } from "./projectApi";
import { editorApi } from "./editor/editorApi";
import { WebEditorRoute } from "./editor/WebEditorRoute";

const API_URL = import.meta.env.VITE_API_URL ?? "";
const DEFAULT_CHAPTER = "1";
const DEFAULT_WORK_TITLE = "Projeto sem nome";
const queryClient = new QueryClient();

type User = { id: string; email: string; role: string };
type Job = {
  id: string;
  status: string;
  obra: string;
  capitulo: string;
  src_lang: string;
  dst_lang: string;
  mode: string;
  page_count?: number;
  processing_seconds?: number;
  error_code?: string;
  error_message?: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  artifacts?: Artifact[];
};
type Artifact = { id: string; kind: string; filename: string; size: number };
type JobEvent = { stage: string; kind: string; message: string; payload?: Record<string, unknown>; created_at?: string };
type AdminOverview = {
  jobs: Pick<Job, "id" | "obra" | "capitulo" | "status">[];
  workers: { id: string; name: string; status: string; max_concurrent_jobs: number; last_seen_at?: string }[];
  audit_logs: { id: string; action: string; entity_type: string; entity_id: string; created_at?: string }[];
};
type NavItem = { to: string; label: string; icon: typeof LayoutDashboard };
type JobProfile = "auto" | "manual" | "batch";

const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled", "deleted"]);
const MANUAL_EDITOR_POLL_MS = 1000;
const MANUAL_EDITOR_TIMEOUT_MS = 180000;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    credentials: "include",
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: "Falha na API" }));
    throw new Error(detail.detail ?? "Falha na API");
  }
  return response.json();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function waitForManualEditorProject(jobId: string): Promise<string> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < MANUAL_EDITOR_TIMEOUT_MS) {
    const { job } = await api<{ job: Job }>(`/api/jobs/${jobId}`);
    if (job.status === "completed") {
      const project = await api<{ project_id: string }>(`/api/jobs/${jobId}/materialize-project`, { method: "POST" });
      return project.project_id;
    }
    if (TERMINAL_JOB_STATUSES.has(job.status)) {
      throw new Error(job.error_message || "Nao foi possivel preparar o editor manual");
    }
    await sleep(MANUAL_EDITOR_POLL_MS);
  }
  throw new Error("O editor manual ainda esta preparando. Tente novamente em instantes.");
}

function useMe() {
  return useQuery({
    queryKey: ["me"],
    queryFn: () => api<{ user: User }>("/api/auth/me"),
    retry: false,
  });
}

function Shell({ children }: { children: React.ReactNode }) {
  const { data } = useMe();
  const userEmail = data?.user.email ?? "admin@local";
  const userName = userEmail.split("@")[0] || userEmail;
  const creditLabel = "1000 créditos";
  const navItems: NavItem[] = [
    { to: "/dashboard", label: "Início", icon: LayoutDashboard },
    { to: "/projects", label: "Projetos", icon: BookOpen },
    { to: "/settings", label: "Config", icon: Settings },
  ];
  return (
    <div className="app-shell desktop-clone-shell">
      <aside className="sidebar desktop-clone-sidebar">
        <Link className="brand desktop-clone-brand" to="/dashboard" aria-label="TraduzAI Web">
          <img className="desktop-clone-brand-logo" src="/assets/traduzai-logo.svg" alt="TraduzAI" />
        </Link>
        <nav className="nav-list desktop-clone-nav">
          <p className="nav-label">Navegação</p>
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
              <item.icon size={16} strokeWidth={1.8} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="worker-note desktop-clone-status">
          <p className="nav-label">Conta</p>
          <div className="status-line status-user"><span>{userName}</span></div>
          <div className="status-line status-credit"><Coins size={13} /><span>{creditLabel}</span></div>
          <div className="sidebar-links">
            <Link to="/legal">Legal</Link>
            {data?.user.role === "admin" && <Link to="/admin">Admin</Link>}
          </div>
        </div>
      </aside>
      <main className="workspace desktop-clone-workspace">
        <div className="desktop-clone-session">
          <span className="user-pill">{userName} · {creditLabel}</span>
          <LogoutButton />
        </div>
        {children}
      </main>
    </div>
  );
}

function Protected({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useMe();
  if (isLoading) return <div className="center-screen">Carregando</div>;
  if (!data?.user) return <Navigate to="/login" replace />;
  return <Shell>{children}</Shell>;
}

function ProtectedFullScreen({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useMe();
  if (isLoading) return <div className="center-screen">Carregando</div>;
  if (!data?.user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function LogoutButton() {
  const navigate = useNavigate();
  const query = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => api("/api/auth/logout", { method: "POST" }),
    onSettled: () => {
      query.clear();
      navigate("/login");
    },
  });
  return <button className="ghost-button" onClick={() => mutation.mutate()} title="Sair"><LogOut size={15} />Sair</button>;
}

function Login() {
  const navigate = useNavigate();
  const query = useQueryClient();
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const googleError = searchParams.get("error");
  const mutation = useMutation({
    mutationFn: () => api<{ user: User }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
    onSuccess: () => {
      query.invalidateQueries({ queryKey: ["me"] });
      navigate("/dashboard");
    },
  });
  return (
    <AuthScreen
      eyebrow="TraduzAI Web"
      title="Acesse sua conta"
      subtitle="Entre para começar a traduzir com o fluxo visual do TraduzAI."
    >
      <form className="auth-card" onSubmit={(event) => { event.preventDefault(); mutation.mutate(); }}>
        <div className="auth-header">
          <p className="eyebrow">Acesso interno</p>
          <h1>Acesse sua conta</h1>
          <p>Entre com sua conta para abrir o painel e continuar seu projeto.</p>
        </div>
        <label>
          Email
          <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Digite seu email" />
        </label>
        <label>
          Senha
          <div className="auth-password-field">
            <input
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Digite sua senha"
              autoFocus
            />
            <button type="button" className="auth-password-toggle" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "Ocultar senha" : "Mostrar senha"}>
              {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </label>
        <div className="auth-meta-row">
          <span className="auth-helper">Esqueceu sua senha? Entre em contato.</span>
        </div>
        {googleError && <p className="error">Não foi possível entrar com Google. Tente novamente.</p>}
        {mutation.error && <p className="error">{mutation.error.message}</p>}
        <button className="primary auth-submit" disabled={mutation.isPending}>Entrar</button>
        <div className="auth-divider"><span>ou continue com</span></div>
        <GoogleAuthButton label="Continuar com Google" />
        <p className="auth-switch-copy">
          Não tem uma conta? <Link className="auth-inline-link" to="/signup">Criar conta</Link>
        </p>
      </form>
    </AuthScreen>
  );
}

function Signup() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const googleError = searchParams.get("error");
  const signupMutation = useMutation({
    mutationFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      return true;
    },
    onSuccess: () => {
      navigate("/login", { state: { signupRequested: true, email } });
    },
  });
  return (
    <AuthScreen
      eyebrow="TraduzAI Web"
      title="Crie sua conta"
      subtitle="Comece a preparar seus capítulos com o fluxo visual do TraduzAI."
    >
      <form className="auth-card" onSubmit={(event) => { event.preventDefault(); signupMutation.mutate(); }}>
        <div className="auth-header">
          <p className="eyebrow">Pré-cadastro</p>
          <h1>Crie sua conta</h1>
          <p>Entre na fila do beta interno e prepare seu acesso ao painel web.</p>
        </div>
        <GoogleAuthButton label="Continuar com Google" />
        <div className="auth-divider"><span>ou crie com email</span></div>
        <label>
          Email
          <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Digite seu email" />
        </label>
        <label>
          Senha
          <div className="auth-password-field">
            <input
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Digite sua senha"
            />
            <button type="button" className="auth-password-toggle" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "Ocultar senha" : "Mostrar senha"}>
              {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </label>
        <label className="auth-checkbox-row">
          <input type="checkbox" checked={acceptedTerms} onChange={(event) => setAcceptedTerms(event.target.checked)} />
          <span>Eu aceito os Termos de Uso e a Política de Privacidade</span>
        </label>
        {googleError && <p className="error">Não foi possível continuar com Google. Tente novamente.</p>}
        <p className="auth-note">Nesta build interna, o cadastro registra seu interesse e redireciona você para o acesso.</p>
        <button className="primary auth-submit" disabled={signupMutation.isPending || !acceptedTerms || !email || !password}>Criar conta</button>
        <p className="auth-switch-copy">
          Já tem uma conta? <Link className="auth-inline-link" to="/login">Fazer login</Link>
        </p>
      </form>
    </AuthScreen>
  );
}

function GoogleAuthButton({ label }: { label: string }) {
  return (
    <a className="auth-secondary-button" href={`${API_URL}/api/auth/google/start?next=/dashboard`}>
      <span className="auth-google-mark">G</span>
      <span>{label}</span>
    </a>
  );
}

function AuthScreen({
  eyebrow,
  title,
  subtitle,
  children,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <main className="auth-screen">
      <div className="auth-screen-bg auth-screen-bg-left" />
      <div className="auth-screen-bg auth-screen-bg-right" />
      <header className="auth-topbar">
        <Link className="auth-brand" to="/">
          <img src="/assets/traduzai-logo.svg" alt="TraduzAI" />
        </Link>
      </header>
      <section className="auth-layout">
        <div className="auth-intro">
          <p className="eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        {children}
      </section>
    </main>
  );
}

function Landing() {
  useEffect(() => {
    const targets = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (!targets.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -14% 0px", threshold: 0.14 },
    );
    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, []);

  const workflow = [
    { title: "Envie o capítulo", body: "Faça upload das páginas em imagem, ZIP ou CBZ.", icon: UploadCloud },
    { title: "Processe com IA", body: "O TraduzAI detecta textos, faz OCR, traduz e limpa a arte.", icon: Sparkles },
    { title: "Revise no editor", body: "Ajuste textos, corrija detalhes e confira o resultado.", icon: LayoutDashboard },
    { title: "Exporte", body: "Baixe o capítulo finalizado em um pacote pronto.", icon: Download },
  ];
  const features = [
    { title: "OCR automático", body: "Extrai textos das páginas.", icon: FileText },
    { title: "Tradução com contexto", body: "Mantém nomes, termos e estilo da obra.", icon: Sparkles },
    { title: "Inpaint inteligente", body: "Remove textos antigos da imagem.", icon: Server },
    { title: "Editor visual", body: "Revise e ajuste antes de exportar.", icon: LayoutDashboard },
    { title: "Exportação CBZ/ZIP", body: "Baixe o capítulo pronto.", icon: Download },
  ];

  return (
    <main className="landing-page">
      <nav className="landing-nav" aria-label="Navegação principal">
        <Link className="landing-logo" to="/">
          <img src="/assets/traduzai-logo.svg" alt="TraduzAI" />
        </Link>
        <div className="landing-nav-links">
          <a href="#fluxo">Fluxo</a>
          <a href="#recursos">Recursos</a>
          <a href="#resultado">Demonstração</a>
          <a href="#faq">FAQ</a>
          <a href="#plano">Planos</a>
        </div>
        <div className="landing-nav-actions">
          <Link className="landing-nav-action" to="/login">Entrar</Link>
          <Link className="landing-nav-action landing-nav-action-primary" to="/signup">Criar conta</Link>
        </div>
      </nav>

      <section className="landing-hero">
        <div className="landing-hero-copy">
          <div className="hero-kicker hero-reveal reveal-delay-1">
            <span>100% local · suas páginas não saem do PC</span>
          </div>
          <h1 className="hero-reveal reveal-delay-2">Traduza um capítulo inteiro de mangá em minutos.</h1>
          <p className="hero-reveal reveal-delay-3">
            OCR, tradução com contexto da obra, inpaint de balões e typesetting — automático. Você só revisa.
          </p>
          <div className="hero-actions-group hero-reveal reveal-delay-4">
            <p className="hero-free-badge">40 páginas grátis para começar</p>
            <div className="landing-actions">
              <Link className="landing-cta primary" to="/signup">Criar conta grátis</Link>
              <a className="landing-cta secondary" href="#resultado">Ver demonstração</a>
            </div>
          </div>
          <div className="hero-flow-preview hero-reveal reveal-delay-5" aria-label="Prévia visual do fluxo TraduzAI">
            <div className="flow-panel">
              <span>Original</span>
              <div className="manga-page manga-original">
                <b>HEY!</b>
                <i />
                <i />
              </div>
            </div>
            <div className="flow-panel flow-processing">
              <span>Processando</span>
              <strong>OCR</strong>
              <small>tradução + limpeza</small>
            </div>
            <div className="flow-panel">
              <span>Final</span>
              <div className="manga-page manga-final">
                <b>EI!</b>
                <i />
                <i />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="fluxo" className="landing-section" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Fluxo</p>
          <h2>Do upload ao capítulo final.</h2>
        </div>
        <div className="workflow-grid">
          {workflow.map((item, index) => (
            <article className="workflow-card" style={{ "--reveal-delay": `${index * 90}ms` } as React.CSSProperties} data-reveal key={item.title}>
              <span className="workflow-index">{index + 1}</span>
              <item.icon size={20} />
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="recursos" className="landing-section feature-band" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Recursos</p>
          <h2>Tudo em um só lugar.</h2>
        </div>
        <div className="feature-grid">
          {features.map((item, index) => (
            <article className="feature-card" style={{ "--reveal-delay": `${index * 110}ms` } as React.CSSProperties} data-reveal key={item.title}>
              <item.icon size={20} />
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="resultado" className="landing-section result-band" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Antes e depois</p>
          <h2>Veja o resultado.</h2>
          <p>Página original em inglês → capítulo traduzido, limpo e pronto para distribuição.</p>
        </div>
        <img
          className="ba-real-image"
          src="/assets/before-after-demo.png"
          alt="Comparação antes e depois: página original em inglês e versão traduzida para português pelo TraduzAI"
          loading="lazy"
        />
      </section>

      <section className="landing-section" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Editor completo</p>
          <h2>Revise cada detalhe antes de exportar.</h2>
          <p>Glossário inteligente, QA automático e editor visual com camadas — tudo em um fluxo integrado.</p>
        </div>
        <div className="screenshot-showcase">
          <img
            src="/assets/editor-view.png"
            alt="Editor visual do TraduzAI com camadas, propriedades de texto e preview da página"
            loading="lazy"
          />
          <div className="screenshot-side">
            <img src="/assets/glossary-panel.png" alt="Painel de glossário inteligente com termos revisados e candidatos" loading="lazy" />
            <img src="/assets/qa-report.png" alt="Relatório de QA com flags de inglês restante e alertas de revisão" loading="lazy" />
          </div>
        </div>
      </section>

      <section id="publico" className="landing-section audience-band" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Para quem é</p>
          <h2>Feito para tradução visual.</h2>
          <p>Para tradutores, editores e equipes que trabalham com mangás, manhwas, webtoons e obras visuais autorizadas.</p>
        </div>
        <div className="audience-grid">
          <article className="audience-card" data-reveal>
            <Languages size={22} />
            <h3>Tradutores</h3>
            <p>Automatize OCR e tradução e foque na revisão criativa. Glossário e memória garantem consistência entre capítulos.</p>
          </article>
          <article className="audience-card" data-reveal>
            <BookOpen size={22} />
            <h3>Equipes de scan</h3>
            <p>Fluxo integrado do upload ao CBZ: contexto da obra, inpaint, typesetting e pacote de revisão em um projeto editável.</p>
          </article>
          <article className="audience-card" data-reveal>
            <Star size={22} />
            <h3>Criadores independentes</h3>
            <p>Lance seus quadrinhos em português sem precisar de equipe grande. Do upload ao capítulo pronto em minutos.</p>
          </article>
        </div>
      </section>

      <section id="faq" className="landing-section" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Dúvidas frequentes</p>
          <h2>Perguntas e respostas.</h2>
        </div>
        <div className="faq-list">
          <div className="faq-item">
            <strong>O TraduzAI fornece mangás?</strong>
            <p>Não. O app edita arquivos do próprio usuário e não hospeda nem distribui obras.</p>
          </div>
          <div className="faq-item">
            <strong>Minhas páginas são enviadas para a internet?</strong>
            <p>Não. As imagens ficam no seu computador. A internet é usada apenas para busca de contexto textual e tradução quando o usuário ativa esses recursos.</p>
          </div>
          <div className="faq-item">
            <strong>Posso revisar antes de exportar?</strong>
            <p>Sim. O fluxo inclui glossário, memória de obra, editor visual, QA automático e múltiplos modos de exportação.</p>
          </div>
          <div className="faq-item">
            <strong>Funciona em qual sistema operacional?</strong>
            <p>Windows 10 e 11 são suportados nesta versão. macOS e Linux estão no roadmap.</p>
          </div>
        </div>
      </section>

      <section id="plano" className="landing-section" data-reveal>
        <div className="landing-section-head">
          <p className="eyebrow">Planos</p>
          <h2>Comece grátis.</h2>
          <p>40 páginas gratuitas para testar. Sem cartão de crédito.</p>
        </div>
        <div className="pricing-grid">
          <article className="pricing-card">
            <p className="pricing-card-name">Free</p>
            <p className="pricing-card-price">Grátis</p>
            <ul className="pricing-card-items">
              <li>40 páginas para testar</li>
              <li>Modo automático e manual</li>
              <li>Export básico (CBZ/ZIP)</li>
              <li>Processamento local</li>
            </ul>
            <Link className="landing-cta primary pricing-card-cta" to="/signup">Criar conta grátis</Link>
          </article>
          <article className="pricing-card featured">
            <p className="pricing-card-name">Pro</p>
            <p className="pricing-card-price">Em desenvolvimento</p>
            <ul className="pricing-card-items">
              <li>Tradução com API própria</li>
              <li>Contexto online de obras</li>
              <li>QA completo</li>
              <li>Glossário e memória</li>
              <li>Export avançado</li>
            </ul>
          </article>
          <article className="pricing-card">
            <p className="pricing-card-name">Studio</p>
            <p className="pricing-card-price">Futuro</p>
            <ul className="pricing-card-items">
              <li>Tradução em lote</li>
              <li>Presets de projeto</li>
              <li>Relatórios de produção</li>
              <li>Pacotes de revisão</li>
              <li>Recursos avançados de equipe</li>
            </ul>
          </article>
        </div>
      </section>

      <footer className="landing-footer">
        <strong>TraduzAI</strong>
        <span>Tradução visual assistida por IA.</span>
        <div>
          <a href="#recursos">Recursos</a>
          <a href="#resultado">Demonstração</a>
          <a href="#faq">FAQ</a>
          <a href="#plano">Planos</a>
          <a href="mailto:contato@traduzai.app">Contato</a>
          <Link to="/legal">Termos</Link>
        </div>
        <small style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
          © {new Date().getFullYear()} TraduzAI
        </small>
      </footer>
    </main>
  );
}

function DashboardEntry() {
  const navigate = useNavigate();
  const [driveLink, setDriveLink] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [pendingDeleteJob, setPendingDeleteJob] = useState<Job | null>(null);
  const query = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["jobs"], queryFn: () => api<{ jobs: Job[] }>("/api/jobs") });
  const jobs = data?.jobs ?? [];
  const recentJobs = jobs.slice(0, 4);
  const deleteJob = useMutation({
    mutationFn: (job: Job) => api(`/api/jobs/${job.id}`, { method: "DELETE" }),
    onSuccess: () => {
      setPendingDeleteJob(null);
      query.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0 });
  }, []);
  const openSetup = (profile: WebProjectMode = "auto") => {
    navigate(`/projects/new?profile=${profile}`, {
      state: {
        driveLink: driveLink.trim(),
        initialFile: file,
        profile,
      },
    });
  };
  const openProjects = () => navigate("/projects");
  const handleDeleteJob = (job: Job) => {
    setPendingDeleteJob(job);
  };
  return (
    <div className="home-dashboard desktop-home-screen dashboard-entry-screen">
      <header className="dashboard-entry-hero">
        <h1>Um capítulo em <em>instantes</em></h1>
        <div className="dashboard-entry-shell">
          <div className="dashboard-link-field">
            <LinkIcon size={17} />
            <input
              value={driveLink}
              onChange={(event) => {
                setDriveLink(event.target.value);
                if (event.target.value.trim()) setFile(null);
              }}
              placeholder="Cole um link do Google Drive, .cbz, .zip ou imagem"
            />
          </div>
          <label className="dashboard-upload-drop">
            <UploadCloud size={19} />
            <strong>{file ? file.name : "Envie seu arquivo"}</strong>
            <span>CBZ, ZIP, PNG, JPG ou WEBP</span>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.zip,.cbz"
              onChange={(event) => {
                const nextFile = event.target.files?.[0] ?? null;
                setFile(nextFile);
                if (nextFile) setDriveLink("");
              }}
            />
          </label>
          <button type="button" className="dashboard-primary-action" onClick={() => openSetup("auto")}>
            <Sparkles size={18} />
            Traduzir capítulo
          </button>
        </div>
      </header>

      <section className="dashboard-tool-section">
        <p className="eyebrow">Ferramentas</p>
        <div className="dashboard-tool-grid">
          <button type="button" className="dashboard-tool-card" onClick={() => openSetup("auto")}>
            <span>
              <strong>Tradução automática</strong>
              <small>OCR, tradução e typesetting com IA local</small>
            </span>
            <Sparkles size={30} />
          </button>
          <button type="button" className="dashboard-tool-card" onClick={() => openSetup("manual")}>
            <span>
              <strong>Editor manual</strong>
              <small>Prepare o projeto e revise cada página</small>
            </span>
            <MousePointer2 size={30} />
          </button>
          <button type="button" className="dashboard-tool-card" onClick={() => openSetup("batch")}>
            <span>
              <strong>Tradução em lote</strong>
              <small>Processe múltiplos capítulos em sequência</small>
            </span>
            <Archive size={30} />
          </button>
          <button type="button" className="dashboard-tool-card" onClick={openProjects}>
            <span>
              <strong>Abrir projeto</strong>
              <small>Importe um ZIP completo ou continue uma tradução</small>
            </span>
            <BookOpen size={30} />
          </button>
        </div>
      </section>

      <section className="dashboard-last-projects">
        <div className="recent-head">
          <span><Clock size={14} />Últimos projetos</span>
          <Link to="/projects">Ver todos <ArrowRight size={13} /></Link>
        </div>
        {isLoading ? <p className="empty">Carregando projetos.</p> : recentJobs.length ? (
          <div className="recent-grid">
            {recentJobs.map((job) => (
              <RecentJobCard
                key={job.id}
                job={job}
                onDelete={handleDeleteJob}
                deleting={deleteJob.isPending}
              />
            ))}
          </div>
        ) : (
          <div className="dashboard-empty-projects">
            <BookOpen size={42} />
            <p>Nenhum projeto ainda</p>
            <button type="button" onClick={() => openSetup("auto")}>Criar seu primeiro projeto</button>
          </div>
        )}
      </section>
      {pendingDeleteJob && (
        <div className="modal-backdrop" role="presentation">
          <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-project-title">
            <p className="eyebrow">Excluir projeto</p>
            <h2 id="delete-project-title">Excluir permanentemente?</h2>
            <p>
              Isso vai excluir permanentemente o projeto <strong>{pendingDeleteJob.obra || "Sem nome"}</strong> e os resultados associados.
              Essa ação não pode ser desfeita.
            </p>
            <div className="confirm-actions">
              <button type="button" onClick={() => setPendingDeleteJob(null)} disabled={deleteJob.isPending}>Cancelar</button>
              <button type="button" className="danger-button" onClick={() => deleteJob.mutate(pendingDeleteJob)} disabled={deleteJob.isPending}>
                Excluir permanentemente
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Dashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["jobs"], queryFn: () => api<{ jobs: Job[] }>("/api/jobs") });
  const jobs = data?.jobs ?? [];
  const recentJobs = jobs.slice(0, 3);
  const freeRemaining = 40;
  const paidCredits = 1000;
  const quota = freeRemaining + paidCredits;
  const progress = Math.max(4, Math.min(100, (freeRemaining / 40) * 100));
  return (
    <div className="home-dashboard desktop-home-screen">
      <header className="hero-block desktop-home-header">
        <div className="hero-kicker">
          <span><Sparkles size={12} /> BETA</span>
          <small>v0.2 · web</small>
        </div>
        <h1>Bem-vindo ao <strong>TraduzAi</strong></h1>
        <p>Traduza mangá, manhwa e manhua automaticamente com IA 100% local.</p>
        <Link className="soft-button" to="/legal">Ajuda</Link>
      </header>

      <section className="quota-banner desktop-quota-card">
        <div className="quota-copy">
          <div className="quota-title"><Sparkles size={14} />Plano gratuito</div>
          <p>{freeRemaining} páginas restantes esta semana · reseta toda segunda-feira</p>
          <div className="quota-track"><span style={{ width: `${progress}%` }} /></div>
          <small>+ {paidCredits} créditos pagos disponíveis</small>
        </div>
        <div className="quota-number">
          <strong>{quota}</strong>
          <span>páginas</span>
        </div>
      </section>

      <section className="desktop-actions desktop-home-actions" aria-label="Ações principais">
        <ActionCard
          to="/projects/new"
          icon={Plus}
          title="Nova tradução"
          description="Escolha automático, manual ou lote"
          primary
        />
      </section>

      <section className="recent-section desktop-recent-section">
        <div className="recent-head">
          <span><Clock size={14} />Projetos recentes</span>
          <small>{recentJobs.length}</small>
        </div>
        {isLoading ? (
          <p className="empty">Carregando projetos recentes.</p>
        ) : recentJobs.length ? (
          <div className="recent-grid">
            {recentJobs.map((job) => <RecentJobCard key={job.id} job={job} />)}
          </div>
        ) : (
          <p className="empty">Nenhum projeto recente ainda.</p>
        )}
      </section>

      <section id="fila" className="panel queue-panel desktop-queue-panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">Fila</p>
            <h2>Jobs</h2>
          </div>
          <Link className="primary link-button" to="/projects/new?profile=auto"><Plus size={16} />Novo job</Link>
        </div>
        {isLoading ? <p>Carregando jobs</p> : <JobTable jobs={jobs} />}
      </section>
    </div>
  );
}

function ActionCard({
  to,
  icon: Icon,
  title,
  description,
  primary = false,
  accent = "brand",
}: {
  to: string;
  icon: typeof Plus;
  title: string;
  description: string;
  primary?: boolean;
  accent?: "brand" | "cyan";
}) {
  return (
    <Link className={`action-card ${primary ? "primary-card" : ""} accent-${accent}`} to={to}>
      <span className="action-icon"><Icon size={24} /></span>
      <span className="action-text">
        <strong>{title}</strong>
        <small>{description}</small>
      </span>
      <ArrowRight className="action-arrow" size={20} />
    </Link>
  );
}

function RecentJobCard({ job, onDelete, deleting = false }: { job: Job; onDelete?: (job: Job) => void; deleting?: boolean }) {
  return (
    <article className="recent-card">
      {onDelete && (
        <button
          type="button"
          className="recent-delete-button"
          onClick={() => onDelete(job)}
          disabled={deleting}
          aria-label={`Excluir permanentemente ${job.obra || "projeto"}`}
          title="Excluir projeto"
        >
          ×
        </button>
      )}
      <Link className="recent-card-link" to={`/job/${job.id}`}>
        <span className="recent-meta"><BookOpen size={14} />Cap. {job.capitulo}</span>
        <strong>{job.obra || "Sem nome"}</strong>
        <span className="recent-pages">
          <CheckCircle2 size={13} />
          {job.page_count ?? "-"} páginas
        </span>
      </Link>
    </article>
  );
}

function JobTable({ jobs }: { jobs: Pick<Job, "id" | "obra" | "capitulo" | "status" | "page_count">[] }) {
  if (!jobs.length) return <p className="empty">Nenhum job criado ainda.</p>;
  return (
    <div className="table">
      <div className="table-row table-head">
        <span>Obra</span>
        <span>Capítulo</span>
        <span>Status</span>
        <span>Páginas</span>
        <span></span>
      </div>
      {jobs.map((job) => (
        <div className="table-row job-data-row" key={job.id}>
          <span data-label="Obra">{job.obra}</span>
          <span data-label="Capítulo">{job.capitulo}</span>
          <span data-label="Status"><Status value={job.status} /></span>
          <span data-label="Páginas">{job.page_count ?? "-"}</span>
          <span data-label="Ação"><Link className="table-link" to={`/job/${job.id}`}>Abrir</Link></span>
        </div>
      ))}
    </div>
  );
}

function ProjectsPage() {
  const { data, isLoading } = useQuery({ queryKey: ["jobs"], queryFn: () => api<{ jobs: Job[] }>("/api/jobs") });
  const jobs = data?.jobs ?? [];
  return (
    <section className="panel desktop-projects-screen">
      <div className="section-head">
        <div>
          <p className="eyebrow">Projetos</p>
          <h1>Traduções recentes</h1>
        </div>
        <Link className="primary link-button" to="/projects/new?profile=auto"><Plus size={16} />Novo projeto</Link>
      </div>
      {isLoading ? <p>Carregando projetos</p> : <JobTable jobs={jobs} />}
    </section>
  );
}

function Status({ value }: { value: string }) {
  return <span className={`status status-${value}`}>{value}</span>;
}

function NewJob() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedProfile = searchParams.get("profile");
  const initialProfile: JobProfile = requestedProfile === "manual" || requestedProfile === "batch" ? requestedProfile : "auto";
  const [file, setFile] = useState<File | null>(null);
  const [obra, setObra] = useState("");
  const [capitulo] = useState(DEFAULT_CHAPTER);
  const [profile, setProfile] = useState<JobProfile>(initialProfile);
  const [mode, setMode] = useState("real");
  const profileCopy = {
    auto: {
      eyebrow: "Novo capítulo",
      title: "Tradução automática",
      help: "Envie um capítulo para OCR, tradução, limpeza e typesetting pelo worker local.",
      accept: ".png,.jpg,.jpeg,.webp,.zip,.cbz",
    },
    manual: {
      eyebrow: "Tradução Manual",
      title: "Preparar para controle manual",
      help: "Use quando quiser preparar o material e revisar o resultado com mais controle depois.",
      accept: ".png,.jpg,.jpeg,.webp,.zip,.cbz",
    },
    batch: {
      eyebrow: "Lote",
      title: "Tradução em lote",
      help: "Envie um ZIP/CBZ com vários capítulos ou páginas para processar em sequência.",
      accept: ".zip,.cbz",
    },
  } satisfies Record<JobProfile, { eyebrow: string; title: string; help: string; accept: string }>;
  const selectedCopy = profileCopy[profile];
  const updateProfile = (next: JobProfile) => {
    setProfile(next);
    setSearchParams({ profile: next });
  };
  const mutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Selecione um arquivo");
      const workTitle = obra.trim() || DEFAULT_WORK_TITLE;
      const form = new FormData();
      form.set("obra", workTitle);
      form.set("capitulo", capitulo.trim() || DEFAULT_CHAPTER);
      form.set("src_lang", "en");
      form.set("dst_lang", "pt-BR");
      form.set("mode", mode);
      form.set("profile", profile);
      form.set("file", file);
      return api<{ job: { id: string } }>("/api/jobs", { method: "POST", body: form });
    },
    onSuccess: (data) => navigate(`/job/${data.job.id}`),
  });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    mutation.mutate();
  };
  return (
    <section className="panel narrow">
      <p className="eyebrow">{selectedCopy.eyebrow}</p>
      <h1>{selectedCopy.title}</h1>
      <p>{selectedCopy.help}</p>
      <form className="form-grid" onSubmit={submit}>
        <div className="segmented-control" aria-label="Tipo de tradução">
          <button type="button" className={profile === "auto" ? "active" : ""} onClick={() => updateProfile("auto")}>Automático</button>
          <button type="button" className={profile === "manual" ? "active" : ""} onClick={() => updateProfile("manual")}>Manual</button>
          <button type="button" className={profile === "batch" ? "active" : ""} onClick={() => updateProfile("batch")}>Lote</button>
        </div>
        <label>Obra <span>(opcional)</span><input value={obra} onChange={(event) => setObra(event.target.value)} /></label>
        <label>Execução<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="real">Worker real</option><option value="mock">Teste rápido</option></select></label>
        <label>Arquivo<input type="file" accept={selectedCopy.accept} onChange={(event) => setFile(event.target.files?.[0] ?? null)} required /></label>
        {mutation.error && <p className="error">{mutation.error.message}</p>}
        <button className="primary" disabled={mutation.isPending}><UploadCloud size={16} />Enviar para fila</button>
      </form>
    </section>
  );
}

function ProjectSetup() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const dashboardState = location.state as { initialFile?: File; driveLink?: string; profile?: WebProjectMode } | null;
  const requestedProfile = dashboardState?.profile ?? searchParams.get("profile");
  const initialMode: WebProjectMode = requestedProfile === "manual" || requestedProfile === "batch" ? requestedProfile : "auto";
  const [mode, setMode] = useState<WebProjectMode>(initialMode);
  const [file, setFile] = useState<File | null>(() => dashboardState?.initialFile ?? null);
  const [driveLink, setDriveLink] = useState(() => dashboardState?.driveLink ?? searchParams.get("drive_link") ?? "");
  const hasDashboardSource = Boolean(dashboardState?.initialFile || dashboardState?.driveLink?.trim());
  const [obra, setObra] = useState("");
  const [capitulo] = useState(DEFAULT_CHAPTER);
  const [srcLang, setSrcLang] = useState("en");
  const [dstLang, setDstLang] = useState("pt-BR");
  const presetId = "scan-clean";
  const [quality, setQuality] = useState<WebProjectConfig["qualidade"]>("normal");
  const [exportMode, setExportMode] = useState<WebProjectConfig["export_mode"]>("clean");
  const [workQuery, setWorkQuery] = useState("");
  const [selectedWork, setSelectedWork] = useState<WorkSearchResult | null>(null);
  const [context, setContext] = useState<WebProjectConfig["contexto"]>(emptyProjectConfig(mode).contexto);
  const [accepted, setAccepted] = useState<GlossaryCandidate[]>([]);
  const [rejected, setRejected] = useState<string[]>([]);
  const [favoriteWorks, setFavoriteWorks] = useState<string[]>(() => {
    try {
      const value = window.localStorage.getItem("traduzai_favorite_works");
      return value ? JSON.parse(value) : [];
    } catch {
      return [];
    }
  });
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [ignoredContextWarning, setIgnoredContextWarning] = useState(false);
  const [warning, setWarning] = useState("");
  const languages = useQuery({ queryKey: ["setup-languages"], queryFn: setupApi.languages });
  const presets = useQuery({ queryKey: ["setup-presets"], queryFn: setupApi.presets });
  const search = useMutation({ mutationFn: setupApi.searchWork });
  const enrich = useMutation({
    mutationFn: (work: WorkSearchResult) => setupApi.workContext(work),
    onSuccess: (data, work) => {
      setSelectedWork(work);
      setContext(data.context);
      setObra(work.title);
      setWorkQuery(work.title);
      setAccepted([]);
      setRejected([]);
      setIgnoredContextWarning(false);
      setWarning("");
    },
  });
  const create = useMutation({
    mutationFn: async () => {
      if (!file && !driveLink.trim()) throw new Error("Selecione o arquivo do capítulo ou informe um link do Google Drive");
      const workTitle = obra.trim() || selectedWork?.title || DEFAULT_WORK_TITLE;
      const chapterNumber = capitulo.trim() || DEFAULT_CHAPTER;
      if ((obra.trim() || workQuery.trim()) && !selectedWork && !ignoredContextWarning) {
        setWarning("Escolha uma obra antes de iniciar ou continue sem contexto.");
        throw new Error("Contexto da obra não revisado");
      }
      const preset = (presets.data?.presets ?? []).find((item) => item.id === presetId);
      const config: WebProjectConfig = {
        ...emptyProjectConfig(mode),
        mode,
        obra: workTitle,
        capitulo: chapterNumber,
        idioma_origem: srcLang,
        idioma_destino: dstLang,
        preset_id: presetId,
        preset,
        qualidade: quality,
        export_mode: exportMode,
        contexto: {
          ...context,
          glossario: Object.fromEntries(accepted.map((item) => [item.source, item.target])),
          internet_context: {
            ...context.internet_context,
            rejected_glossary_candidates: rejected,
          },
        },
        work_context: selectedWork
          ? {
              selected: true,
              work_id: selectedWork.work_id,
              title: selectedWork.title,
              context_loaded: true,
              internet_context_loaded: Boolean(context.internet_context?.internet_context_loaded),
              glossary_loaded: true,
              glossary_entries_count: accepted.length,
              risk_level: selectedWork.risk_level,
              user_ignored_warning: ignoredContextWarning,
            }
          : {
              selected: false,
              work_id: "",
              title: "",
              context_loaded: false,
              internet_context_loaded: false,
              glossary_loaded: false,
              glossary_entries_count: accepted.length,
              risk_level: "high",
              user_ignored_warning: true,
            },
      };
      const form = new FormData();
      form.set("obra", workTitle);
      form.set("capitulo", chapterNumber);
      form.set("src_lang", srcLang);
      form.set("dst_lang", dstLang);
      form.set("mode", "real");
      form.set("project_config", JSON.stringify(config));
      if (file) form.set("file", file);
      if (!file && driveLink.trim()) form.set("drive_link", driveLink.trim());
      const result = await api<{ job: { id: string } }>("/api/jobs", { method: "POST", body: form });
      if (mode !== "manual") {
        return { job: result.job, editorProjectId: null };
      }
      const editorProjectId = await waitForManualEditorProject(result.job.id);
      return { job: result.job, editorProjectId };
    },
    onSuccess: (data) => {
      if (data.editorProjectId) {
        navigate(`/projects/${data.editorProjectId}/editor?page=0`);
        return;
      }
      navigate(`/job/${data.job.id}`);
    },
  });
  const importProject = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Selecione o ZIP completo exportado");
      const form = new FormData();
      form.set("file", file);
      const result = await api<{ project_id: string }>("/api/projects/import", { method: "POST", body: form });
      return result;
    },
    onSuccess: (data) => navigate(`/projects/${data.project_id}/preview`),
  });
  const normalizedWorkQuery = workQuery.trim().toLocaleLowerCase("pt-BR");
  const isFavoriteWork = Boolean(normalizedWorkQuery) && favoriteWorks.some((item) => item.toLocaleLowerCase("pt-BR") === normalizedWorkQuery);
  const favoriteSuggestions = favoriteWorks
    .filter((item) => !normalizedWorkQuery || item.toLocaleLowerCase("pt-BR").includes(normalizedWorkQuery))
    .slice(0, 5);
  const searchResults = search.data?.results ?? [];
  const displayedWorkResults = selectedWork ? [selectedWork] : searchResults;
  const selectedSourceLabel = file
    ? file.name
    : driveLink.trim()
      ? hasDashboardSource ? "Fonte selecionada no dashboard" : "Google Drive: link informado"
      : "Arquivo não selecionado";

  const setupStartLabel = mode === "manual" ? "Começar a editar" : "Enviar para fila";
  const setupPendingLabel = mode === "manual" ? "Preparando editor..." : "Enviando...";

  const persistFavoriteWorks = (next: string[]) => {
    setFavoriteWorks(next);
    window.localStorage.setItem("traduzai_favorite_works", JSON.stringify(next));
  };

  const toggleFavoriteWork = () => {
    const title = workQuery.trim();
    if (!title) return;
    if (isFavoriteWork) {
      persistFavoriteWorks(favoriteWorks.filter((item) => item.toLocaleLowerCase("pt-BR") !== normalizedWorkQuery));
      return;
    }
    persistFavoriteWorks([title, ...favoriteWorks.filter((item) => item.toLocaleLowerCase("pt-BR") !== normalizedWorkQuery)].slice(0, 12));
  };

  const runWorkSearch = (query = workQuery) => {
    const value = query.trim();
    if (!value) return;
    setSelectedWork(null);
    setContext(emptyProjectConfig(mode).contexto);
    setAccepted([]);
    setRejected([]);
    setIgnoredContextWarning(false);
    setWarning("");
    search.reset();
    enrich.reset();
    setWorkQuery(value);
    setObra(value);
    search.mutate(value);
  };

  return (
    <section className="setup-page desktop-setup-screen">
      <div className="setup-header desktop-setup-header desktop-setup-titlebar">
        <Link className="desktop-setup-back" to="/dashboard">
          <ArrowLeft size={14} />
          Configurar projeto
        </Link>
        <div className="desktop-setup-heading">
          <h1>Nova tradução</h1>
          <p>{mode === "manual" ? "Modo manual prepara o projeto e abre o editor para controle total." : "Configure contexto e idioma antes de enviar ao worker local."}</p>
        </div>
      </div>
      <form className="setup-grid desktop-setup-flow" onSubmit={(event) => { event.preventDefault(); create.mutate(); }}>
        <div className="desktop-setup-mode-switch" aria-label="Tipo de tradução">
          <div className="segmented-control">
            <button type="button" className={mode === "auto" ? "active" : ""} onClick={() => setMode("auto")}>Auto</button>
            <button type="button" className={mode === "manual" ? "active" : ""} onClick={() => setMode("manual")}>Manual</button>
            <button type="button" className={mode === "batch" ? "active" : ""} onClick={() => setMode("batch")}>Lote</button>
          </div>
        </div>
        <section className="setup-panel desktop-work-search-panel desktop-setup-main-panel" data-testid="work-context-summary">
          <div className="desktop-setup-context-column">
            <label className="desktop-work-label">Nome da obra <span>(opcional)</span></label>
            <div className="desktop-work-search-row">
              <div className="desktop-work-search-input">
                <input
                  data-testid="project-name-input"
                  value={workQuery}
                  onChange={(event) => {
                    const nextValue = event.target.value;
                    setWorkQuery(nextValue);
                    setObra(nextValue);
                    if (selectedWork && nextValue.trim() !== selectedWork.title) {
                      setSelectedWork(null);
                      setContext(emptyProjectConfig(mode).contexto);
                      setAccepted([]);
                      setRejected([]);
                      setIgnoredContextWarning(false);
                      setWarning("");
                      enrich.reset();
                    }
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      runWorkSearch();
                    }
                  }}
                  placeholder="Ex: Solo Leveling, One Piece..."
                />
                <Search size={16} />
              </div>
              <button
                type="button"
                className={isFavoriteWork ? "desktop-favorite-button active" : "desktop-favorite-button"}
                onClick={toggleFavoriteWork}
                disabled={!workQuery.trim()}
                title={isFavoriteWork ? "Remover das favoritas" : "Adicionar às favoritas"}
              >
                <Star size={17} className={isFavoriteWork ? "fill-current" : ""} />
              </button>
              <button type="button" className="desktop-work-search-button" data-testid="project-search-button" onClick={() => runWorkSearch()} disabled={search.isPending}>
                {search.isPending ? "..." : "Buscar"}
              </button>
            </div>
            {favoriteSuggestions.length > 0 && (
              <div className="desktop-favorite-chips">
                {favoriteSuggestions.map((title) => (
                  <button type="button" key={title} onClick={() => runWorkSearch(title)}>
                    {title}
                  </button>
                ))}
              </div>
            )}
            {favoriteWorks.length > 0 && (
              <p className="desktop-saved-favorites">Favoritas salvas: {favoriteWorks.slice(0, 6).join(", ")}</p>
            )}
            {(search.error || enrich.error) && <p className="error">{(search.error || enrich.error)?.message}</p>}
            <div className="desktop-work-results">
              <div className="desktop-work-results-title">
                <Sparkles size={15} />
                <span>{selectedWork ? "Obra selecionada para montar o contexto" : "Escolha a obra certa para montar o contexto"}</span>
              </div>
              {displayedWorkResults.length ? displayedWorkResults.map((item) => {
                const isSelected = selectedWork?.work_id === item.work_id;
                return (
                <button
                  type="button"
                  key={item.work_id}
                  data-testid="work-result-item"
                  className={isSelected ? "desktop-work-result active" : "desktop-work-result"}
                  onClick={() => {
                    if (!isSelected) enrich.mutate(item);
                  }}
                  disabled={enrich.isPending && !isSelected}
                >
                  <div className="desktop-work-result-head">
                    <div>
                      <strong>{item.title}</strong>
                      <span>{item.source === "anilist" ? "ANILIST" : item.source.toLocaleUpperCase("pt-BR")}</span>
                    </div>
                    <small>{item.source}</small>
                  </div>
                  {item.synopsis && <p>{item.synopsis}</p>}
                  <div className="desktop-work-result-foot">
                    <span>Score {Math.round(item.score ?? 100)}</span>
                    <span>{isSelected ? "Selecionada" : enrich.isPending ? "Carregando..." : "Usar esta obra"}</span>
                  </div>
                </button>
              );
              }) : (
                <div className="desktop-work-empty">
                  {search.isPending ? "Buscando na internet..." : "Busque pelo nome para consultar AniList e montar o contexto."}
                </div>
              )}
            </div>
            <div className="desktop-work-context-state">
              {selectedWork ? `Contexto carregado para ${selectedWork.title}.` : ignoredContextWarning ? "Você escolheu continuar sem contexto." : "Nenhuma obra selecionada."}
              {!selectedWork && (
                <button type="button" data-testid="work-context-continue-without-context" onClick={() => { setIgnoredContextWarning(true); setWarning(""); }}>
                  Continuar sem contexto
                </button>
              )}
            </div>
          </div>

          <div className="desktop-setup-entry-column">
            <p className="eyebrow">Entrada</p>
            <div className="setup-two">
              <label>Origem<select value={srcLang} onChange={(event) => setSrcLang(event.target.value)}>{(languages.data?.languages ?? []).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              <label>Destino<select value={dstLang} onChange={(event) => setDstLang(event.target.value)}>{(languages.data?.languages ?? []).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
            </div>
            {!hasDashboardSource && (
              <div className="desktop-source-grid">
                <label className="desktop-source-card">
                  <span>Upload local</span>
                  <input type="file" accept=".png,.jpg,.jpeg,.webp,.zip,.cbz" onChange={(event) => { setFile(event.target.files?.[0] ?? null); if (event.target.files?.[0]) setDriveLink(""); }} />
                </label>
                <div className="desktop-source-card">
                  <span>Google Drive</span>
                  <input value={driveLink} onChange={(event) => { setDriveLink(event.target.value); if (event.target.value.trim()) setFile(null); }} placeholder="Cole o link do arquivo" />
                  <button type="button" onClick={() => window.open("https://drive.google.com/drive/my-drive", "_blank", "noopener,noreferrer")}>Selecionar no Drive</button>
                </div>
              </div>
            )}
          </div>
        </section>

        <details className="setup-panel" data-testid="setup-advanced-panel" open={advancedOpen} onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}>
          <summary>Avançado</summary>
          <div className="setup-two">
            <label>Qualidade<select value={quality} onChange={(event) => setQuality(event.target.value as WebProjectConfig["qualidade"])}><option value="rapida">Rápida</option><option value="normal">Normal</option><option value="alta">Alta</option></select></label>
            <label>Export padrão<select value={exportMode} onChange={(event) => setExportMode(event.target.value as WebProjectConfig["export_mode"])}><option value="clean">Seguro</option><option value="with_warnings">Com avisos</option><option value="debug">Debug</option></select></label>
          </div>
        </details>

        <section className="setup-panel setup-review">
          <p className="eyebrow">Revisão final</p>
          <p>{selectedSourceLabel}</p>
          <p>{selectedWork ? "Contexto pronto" : ignoredContextWarning ? "Sem contexto por escolha do usuário" : "Contexto pendente"}</p>
          {warning && <p className="error">{warning}</p>}
          {(create.error || importProject.error) && <p className="error">{(create.error || importProject.error)?.message}</p>}
          <button className="primary" data-testid="setup-start-button" disabled={create.isPending}>
            {mode === "manual" ? <MousePointer2 size={16} /> : <UploadCloud size={16} />}
            {create.isPending ? setupPendingLabel : setupStartLabel}
          </button>
          <button type="button" onClick={() => importProject.mutate()} disabled={importProject.isPending}>Importar ZIP completo</button>
        </section>
      </form>
    </section>
  );
}

function ProjectPreview() {
  const { id } = useParams();
  const [pageIndex, setPageIndex] = useState(0);
  const query = useQuery({ queryKey: ["project", id], queryFn: () => projectApi.getProject(id!), enabled: Boolean(id), retry: false });
  const page = useQuery({ queryKey: ["project-page", id, pageIndex], queryFn: () => projectApi.getPage(id!, pageIndex), enabled: Boolean(id) && Boolean(query.data?.project), retry: false });
  const materialize = useMutation({ mutationFn: () => projectApi.materialize(id!), onSuccess: () => query.refetch() });
  const render = useMutation({ mutationFn: () => projectApi.renderPreview(id!, pageIndex), onSuccess: () => page.refetch() });
  const exportMutation = useMutation({
    mutationFn: (format: "zip-full" | "cbz" | "jpg-zip") => projectApi.exportProject(id!, format),
    onSuccess: (data) => {
      window.location.href = assetUrl(data.artifact.download_url);
    },
  });
  const project = query.data?.project;
  const pages = project?.paginas ?? [];
  if (query.error) {
    return (
      <section className="panel narrow">
        <p className="eyebrow">Preview</p>
        <h1>Preparar projeto web</h1>
        <p>Este job ainda não tem workspace web materializado.</p>
        <button className="primary" onClick={() => materialize.mutate()} disabled={materialize.isPending}>Preparar preview</button>
        {materialize.error && <p className="error">{materialize.error.message}</p>}
      </section>
    );
  }
  return (
    <section className="project-preview desktop-preview-screen">
      <div className="preview-header desktop-preview-header">
        <div>
          <p className="eyebrow">Preview</p>
          <h1>{project?.obra ?? "Projeto"}</h1>
          <p>Capítulo {project?.capitulo ?? "-"} · {pages.length} páginas</p>
        </div>
        <div className="action-row">
          <Link className="link-button" to={`/projects/${id}/editor?page=${pageIndex}`}>Editar página</Link>
          <Link className="link-button" to={`/projects/${id}/settings`}>Configurar</Link>
        </div>
      </div>
      <div className="preview-layout desktop-preview-layout">
        <aside className="thumbnail-rail">
          {pages.map((item: any, index: number) => (
            <button key={index} className={index === pageIndex ? "thumb active" : "thumb"} onClick={() => setPageIndex(index)}>
              <span>{index + 1}</span>
              <small>{item.rendered_path || item.translated_path || "página"}</small>
            </button>
          ))}
        </aside>
        <main className="preview-main desktop-preview-canvas">
          <LayerComparison projectId={id!} layers={page.data?.layers ?? {}} state={page.data?.state} pageIndex={pageIndex} />
          <div className="preview-actions">
            <button onClick={() => render.mutate()} disabled={render.isPending}>Renderizar preview</button>
            <button onClick={() => exportMutation.mutate("zip-full")} disabled={exportMutation.isPending}>ZIP completo</button>
            <button onClick={() => exportMutation.mutate("cbz")} disabled={exportMutation.isPending}>CBZ</button>
            <button onClick={() => exportMutation.mutate("jpg-zip")} disabled={exportMutation.isPending}>JPG</button>
          </div>
          {(render.error || exportMutation.error) && <p className="error">{(render.error || exportMutation.error)?.message}</p>}
        </main>
      </div>
    </section>
  );
}

function LayerComparison({ projectId, layers, state, pageIndex }: { projectId: string; layers: ProjectLayerMap; state: any; pageIndex: number }) {
  const preview = state?.preview?.[String(pageIndex)];
  const previewUrl = preview?.asset_path ? assetUrl(`/api/projects/${projectId}/assets/${preview.asset_path}`) : null;
  const items = [
    ["Original", layers.base?.url],
    ["Traduzido", layers.rendered?.url || layers.translated?.url],
    ["Preview fiel", previewUrl],
  ] as const;
  return (
    <div className="comparison-grid">
      {items.map(([label, url]) => (
        <article className="preview-image-panel" key={label}>
          <strong>{label}</strong>
          {url ? <img src={url.startsWith("/api") ? assetUrl(url) : url} alt={label} /> : <div className="empty">Sem imagem</div>}
        </article>
      ))}
      <div className="layer-status">
        {["base", "mask", "inpaint", "brush", "recovery", "rendered"].map((key) => <span key={key} className={layers[key] ? "available" : ""}>{key}</span>)}
      </div>
    </div>
  );
}

function WebEditor() {
  const { id } = useParams();
  const [searchParams] = useSearchParams();
  const pageIndex = Number(searchParams.get("page") ?? 0);
  const navigate = useNavigate();
  const query = useQuery({ queryKey: ["editor-page", id, pageIndex], queryFn: () => editorApi.loadEditorPage(id!, pageIndex), enabled: Boolean(id) });
  const projectQuery = useQuery({ queryKey: ["project", id], queryFn: () => projectApi.getProject(id!), enabled: Boolean(id) });
  const pageAssets = useQuery({ queryKey: ["project-page", id, pageIndex], queryFn: () => projectApi.getPage(id!, pageIndex), enabled: Boolean(id) });
  const [selectedId, setSelectedId] = useState<string>("");
  const [dragging, setDragging] = useState<{ id: string; startX: number; startY: number; baseX: number; baseY: number } | null>(null);
  const [draftPositions, setDraftPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [tool, setTool] = useState<"select" | "text" | "brush" | "mask" | "recovery" | "eraser">("select");
  const [view, setView] = useState<"original" | "translated" | "preview">("translated");
  const [zoom, setZoom] = useState(1);
  const [brushSize, setBrushSize] = useState(28);
  const [brushColor, setBrushColor] = useState("#00d4ff");
  const [painting, setPainting] = useState(false);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const page = query.data?.page;
  const project = projectQuery.data?.project ?? query.data?.project;
  const pages = project?.paginas ?? (page ? [page] : []);
  const layers = (page?.text_layers ?? page?.textos ?? []) as any[];
  const selected = layers.find((item) => String(item.id) === selectedId) ?? layers[0];
  useEffect(() => {
    if (!selectedId && layers[0]?.id) setSelectedId(String(layers[0].id));
  }, [layers, selectedId]);
  const patch = useMutation({
    mutationFn: ({ layerId, data }: { layerId: string; data: any }) => editorApi.patchTextLayer(id!, pageIndex, layerId, data),
    onSuccess: () => query.refetch(),
  });
  const create = useMutation({
    mutationFn: (point?: { x: number; y: number }) => editorApi.createTextLayer(id!, pageIndex, {
      id: `text-${Date.now()}`,
      texto: "Novo texto",
      x: point?.x ?? 80,
      y: point?.y ?? 80,
      w: 220,
      h: 90,
      font_size: 28,
      color: "#ffffff",
      align: "center",
      bold: true,
    }),
    onSuccess: (data) => { setSelectedId(String(data.layer.id)); query.refetch(); },
  });
  const remove = useMutation({
    mutationFn: (layerId: string) => editorApi.deleteTextLayer(id!, pageIndex, layerId),
    onSuccess: () => { setSelectedId(""); query.refetch(); },
  });
  const bitmap = useMutation({
    mutationFn: ({ layer, pngData }: { layer: "mask" | "brush" | "recovery"; pngData?: string }) => editorApi.updateBitmapLayer(id!, pageIndex, layer, {
      png_data: pngData,
      color: brushColor,
      opacity: 0.72,
      hardness: 0.7,
    }),
    onSuccess: () => pageAssets.refetch(),
  });
  const action = useMutation({ mutationFn: (next: string) => editorApi.runPageAction(id!, pageIndex, next), onSuccess: () => query.refetch() });
  const previewAssetPath = pageAssets.data?.state?.preview?.[String(pageIndex)]?.asset_path;
  const imageUrl = (
    view === "original"
      ? pageAssets.data?.layers.base?.url
      : view === "preview"
        ? previewAssetPath
          ? `/api/projects/${id}/assets/${previewAssetPath}`
          : pageAssets.data?.layers.rendered?.url
        : pageAssets.data?.layers.rendered?.url || pageAssets.data?.layers.translated?.url
  ) || pageAssets.data?.layers.base?.url;
  const move = (dx: number, dy: number) => {
    if (!selected) return;
    patch.mutate({ layerId: String(selected.id), data: { x: Number(selected.x ?? selected.bbox?.[0] ?? 0) + dx, y: Number(selected.y ?? selected.bbox?.[1] ?? 0) + dy } });
  };
  const layerPosition = (layer: any) => draftPositions[String(layer.id)] ?? { x: Number(layer.x ?? layer.bbox?.[0] ?? 40), y: Number(layer.y ?? layer.bbox?.[1] ?? 40) };
  const layerSize = (layer: any) => ({ width: Number(layer.w ?? layer.width ?? layer.bbox?.[2] ?? 180), height: Number(layer.h ?? layer.height ?? layer.bbox?.[3] ?? 70) });
  const finishDrag = () => {
    if (!dragging) return;
    const position = draftPositions[dragging.id];
    setDragging(null);
    if (position) patch.mutate({ layerId: dragging.id, data: position });
  };
  const stagePoint = (event: React.MouseEvent<HTMLElement>) => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect || !stageRef.current) return { x: 80, y: 80 };
    return {
      x: Math.round((event.clientX - rect.left + stageRef.current.scrollLeft) / zoom),
      y: Math.round((event.clientY - rect.top + stageRef.current.scrollTop) / zoom),
    };
  };
  const prepareCanvas = () => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return null;
    const width = Math.max(1, Math.round(stage.scrollWidth / zoom));
    const height = Math.max(1, Math.round(stage.scrollHeight / zoom));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return canvas.getContext("2d");
  };
  const paintAt = (event: React.MouseEvent<HTMLElement>) => {
    const ctx = prepareCanvas();
    if (!ctx) return;
    const point = stagePoint(event);
    ctx.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
    ctx.fillStyle = tool === "mask" ? "rgba(124, 92, 255, 0.72)" : tool === "recovery" ? "rgba(74, 222, 128, 0.78)" : brushColor;
    ctx.beginPath();
    ctx.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
  };
  const commitPaint = () => {
    if (!painting) return;
    setPainting(false);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const targetLayer = tool === "mask" ? "mask" : tool === "recovery" ? "recovery" : "brush";
    bitmap.mutate({ layer: targetLayer, pngData: canvas.toDataURL("image/png") });
  };
  const setSelectedStyle = (data: any) => {
    if (!selected) return;
    patch.mutate({ layerId: String(selected.id), data });
  };
  const goPage = (next: number) => {
    if (!id) return;
    navigate(`/projects/${id}/editor?page=${Math.max(0, Math.min(next, Math.max(0, pages.length - 1)))}`);
  };
  const toolItems = [
    { key: "select", icon: MousePointer2, label: "V", title: "Selecionar" },
    { key: "text", icon: FileText, label: "T", title: "Novo texto" },
    { key: "brush", icon: Brush, label: "B", title: "Brush" },
    { key: "recovery", icon: RotateCcw, label: "R", title: "Recovery" },
    { key: "eraser", icon: Eraser, label: "E", title: "Borracha" },
    { key: "mask", icon: Scissors, label: "L", title: "Máscara" },
  ] as const;
  return (
    <section className="web-editor desktop-editor-screen">
      <header className="editor-toolbar desktop-editor-toolbar">
        <Link className="icon-button" to={`/projects/${id}/preview`} title="Voltar ao preview"><ChevronLeft size={16} /></Link>
        <div className="desktop-editor-title">
          <strong>{project?.obra ?? "Projeto"}</strong>
          <span>Cap. {project?.capitulo ?? "-"} · Página {pageIndex + 1}/{Math.max(1, pages.length || 1)}</span>
        </div>
        <div className="desktop-editor-views">
          {[
            ["original", Image, "Original"],
            ["translated", Layers, "Camadas"],
            ["preview", Eye, "Preview"],
          ].map(([key, Icon, label]) => (
            <button key={String(key)} className={view === key ? "active" : ""} onClick={() => setView(key as typeof view)} title={String(label)}>
              <Icon size={13} />{String(label)}
            </button>
          ))}
        </div>
        <div className="desktop-editor-page-nav">
          <button onClick={() => goPage(pageIndex - 1)} disabled={pageIndex <= 0}><ArrowLeft size={13} /></button>
          <span>{pageIndex + 1}/{Math.max(1, pages.length || 1)}</span>
          <button onClick={() => goPage(pageIndex + 1)} disabled={pageIndex >= pages.length - 1}><ArrowRight size={13} /></button>
        </div>
        <div className="desktop-editor-pipeline">
          <button onClick={() => action.mutate("ocr")} disabled={action.isPending}><FileText size={12} />OCR</button>
          <button onClick={() => action.mutate("translate")} disabled={action.isPending}><Languages size={12} />Traduzir</button>
          <button onClick={() => action.mutate("inpaint")} disabled={action.isPending}><Eraser size={12} />Inpaint</button>
        </div>
        <button className="primary desktop-editor-save" onClick={() => projectApi.renderPreview(id!, pageIndex).then(() => pageAssets.refetch())}>
          Renderizar
        </button>
      </header>
      <div className="editor-layout desktop-editor-layout">
        <aside className="desktop-editor-thumbs">
          {pages.map((item: any, index: number) => (
            <button key={index} className={index === pageIndex ? "active" : ""} onClick={() => goPage(index)}>
              <span>{index + 1}</span>
              <small>{item.rendered_path || item.translated_path || item.original_path || "Página"}</small>
            </button>
          ))}
        </aside>
        <aside className="desktop-tool-sidebar">
          {toolItems.map(({ key, icon: Icon, label, title }) => (
            <button key={key} className={tool === key ? "active" : ""} onClick={() => setTool(key)} title={title}>
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
          <div className="desktop-tool-separator" />
          <button onClick={() => setZoom((current) => Math.min(3, current + 0.1))}>+</button>
          <button onClick={() => setZoom(1)}>1x</button>
          <button onClick={() => setZoom((current) => Math.max(0.3, current - 0.1))}>-</button>
        </aside>
        <aside className="editor-side desktop-layer-panel">
          <p className="eyebrow">Camadas</p>
          <button className="layer-row desktop-layer-create" onClick={() => create.mutate(undefined)}>+ Novo texto</button>
          {layers.map((layer, index) => (
            <button key={layer.id ?? `${pageIndex}-${index}`} className={String(layer.id) === String(selected?.id) ? "layer-row active" : "layer-row"} onClick={() => setSelectedId(String(layer.id))}>
              {layer.texto || layer.traduzido || layer.id}
            </button>
          ))}
        </aside>
        <main
          ref={stageRef}
          className={`editor-stage desktop-editor-stage tool-${tool}`}
          onMouseMove={(event) => {
            if (painting) paintAt(event);
            if (!dragging) return;
            setDraftPositions((current) => ({
              ...current,
              [dragging.id]: {
                x: Math.max(0, dragging.baseX + event.clientX - dragging.startX),
                y: Math.max(0, dragging.baseY + event.clientY - dragging.startY),
              },
            }));
          }}
          onMouseUp={finishDrag}
          onMouseLeave={() => { finishDrag(); commitPaint(); }}
          onMouseDown={(event) => {
            if (tool === "text") {
              create.mutate(stagePoint(event));
              setTool("select");
              return;
            }
            if (tool === "brush" || tool === "mask" || tool === "recovery" || tool === "eraser") {
              setPainting(true);
              paintAt(event);
            }
          }}
          onMouseUpCapture={commitPaint}
        >
          <div className="desktop-editor-canvas" style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}>
            {imageUrl ? <img src={assetUrl(imageUrl)} alt="Página" /> : <div className="empty">Sem imagem</div>}
            <canvas ref={canvasRef} className="desktop-paint-canvas" />
            {layers.map((layer) => {
              const size = layerSize(layer);
              return (
                <button
                  type="button"
                  key={layer.id}
                  className={String(layer.id) === String(selected?.id) ? "text-box active" : "text-box"}
                  style={{
                    left: `${layerPosition(layer).x}px`,
                    top: `${layerPosition(layer).y}px`,
                    width: `${size.width}px`,
                    minHeight: `${size.height}px`,
                    color: layer.color ?? "#fff",
                    fontSize: `${Number(layer.font_size ?? layer.fontSize ?? 24)}px`,
                    fontWeight: layer.bold ? 800 : 600,
                    fontStyle: layer.italic ? "italic" : "normal",
                    textAlign: layer.align ?? "center",
                  }}
                  onMouseDown={(event) => {
                    if (tool !== "select") return;
                    event.stopPropagation();
                    const layerId = String(layer.id);
                    const position = layerPosition(layer);
                    setSelectedId(layerId);
                    setDragging({ id: layerId, startX: event.clientX, startY: event.clientY, baseX: position.x, baseY: position.y });
                  }}
                  onClick={(event) => { event.stopPropagation(); setSelectedId(String(layer.id)); }}
                >
                  {layer.texto || layer.traduzido || "Texto"}
                </button>
              );
            })}
          </div>
        </main>
        <aside className="editor-side desktop-properties-panel">
          <p className="eyebrow">Texto</p>
          {selected ? (
            <div className="form-grid">
              <label>Conteúdo<textarea value={selected.texto || selected.traduzido || ""} onChange={(event) => setSelectedStyle({ texto: event.target.value, traduzido: event.target.value })} /></label>
              <div className="desktop-type-row">
                <label>Tamanho<input type="number" value={selected.font_size ?? selected.fontSize ?? 24} onChange={(event) => setSelectedStyle({ font_size: Number(event.target.value) })} /></label>
                <label>Cor<input type="color" value={selected.color ?? "#ffffff"} onChange={(event) => setSelectedStyle({ color: event.target.value })} /></label>
              </div>
              <div className="desktop-inline-tools">
                <button className={selected.align === "left" ? "active" : ""} onClick={() => setSelectedStyle({ align: "left" })}><AlignLeft size={14} /></button>
                <button className={(selected.align ?? "center") === "center" ? "active" : ""} onClick={() => setSelectedStyle({ align: "center" })}><AlignCenter size={14} /></button>
                <button className={selected.align === "right" ? "active" : ""} onClick={() => setSelectedStyle({ align: "right" })}><AlignRight size={14} /></button>
                <button className={selected.bold ? "active" : ""} onClick={() => setSelectedStyle({ bold: !selected.bold })}><Bold size={14} /></button>
                <button className={selected.italic ? "active" : ""} onClick={() => setSelectedStyle({ italic: !selected.italic })}><Italic size={14} /></button>
              </div>
              <div className="nudge-grid">
                <button onClick={() => move(0, -8)}>Cima</button>
                <button onClick={() => move(-8, 0)}>Esq.</button>
                <button onClick={() => move(8, 0)}>Dir.</button>
                <button onClick={() => move(0, 8)}>Baixo</button>
              </div>
              <button className="danger" onClick={() => remove.mutate(String(selected.id))}><Trash2 size={14} />Excluir camada</button>
            </div>
          ) : <p className="empty">Sem texto selecionado.</p>}
          <div className="desktop-brush-panel">
            <p className="eyebrow">Pincel</p>
            <label>Tamanho<input type="range" min={4} max={120} value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} /></label>
            <label>Cor<input type="color" value={brushColor} onChange={(event) => setBrushColor(event.target.value)} /></label>
          </div>
        </aside>
      </div>
      {(query.error || patch.error || action.error || remove.error || bitmap.error) && <p className="error">{(query.error || patch.error || action.error || remove.error || bitmap.error)?.message}</p>}
    </section>
  );
}

function ProjectSettings() {
  const { id } = useParams();
  const query = useQuery({ queryKey: ["project", id], queryFn: () => projectApi.getProject(id!), enabled: Boolean(id) });
  const [draft, setDraft] = useState<any>({});
  useEffect(() => {
    if (query.data?.project) setDraft(query.data.project);
  }, [query.data?.project]);
  const save = useMutation({ mutationFn: () => projectApi.saveSettings(id!, { obra: draft.obra, capitulo: draft.capitulo, idioma_origem: draft.idioma_origem, idioma_destino: draft.idioma_destino, config: draft.config ?? {} }), onSuccess: () => query.refetch() });
  return (
    <section className="panel narrow desktop-project-settings-screen">
      <p className="eyebrow">Configurar projeto</p>
      <h1>Projeto</h1>
      <div className="form-grid">
        <label>Obra<input value={draft.obra ?? ""} onChange={(event) => setDraft({ ...draft, obra: event.target.value })} /></label>
        <label>Capítulo<input value={draft.capitulo ?? ""} onChange={(event) => setDraft({ ...draft, capitulo: event.target.value })} /></label>
        <label>Origem<input value={draft.idioma_origem ?? ""} onChange={(event) => setDraft({ ...draft, idioma_origem: event.target.value })} /></label>
        <label>Destino<input value={draft.idioma_destino ?? ""} onChange={(event) => setDraft({ ...draft, idioma_destino: event.target.value })} /></label>
        <button className="primary" onClick={() => save.mutate()} disabled={save.isPending}>Salvar</button>
      </div>
      {save.error && <p className="error">{save.error.message}</p>}
    </section>
  );
}

function useLiveEvents(jobId: string | undefined) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const query = useQueryClient();
  useEffect(() => {
    if (!jobId) return;
    const source = new EventSource(`${API_URL}/api/jobs/${jobId}/events`, { withCredentials: true });
    source.onmessage = (message) => {
      if (!message.data || message.data === "{}") return;
      setEvents((prev) => [JSON.parse(message.data), ...prev].slice(0, 80));
      query.invalidateQueries({ queryKey: ["job", jobId] });
      query.invalidateQueries({ queryKey: ["jobs"] });
    };
    source.addEventListener("status", source.onmessage);
    source.addEventListener("error", source.onmessage);
    source.addEventListener("artifact", source.onmessage);
    return () => source.close();
  }, [jobId, query]);
  return events;
}

const JOB_PROCESS_STEPS = [
  { id: "extract", label: "Extração", detail: "Descompactando e validando arquivos" },
  { id: "ocr", label: "OCR", detail: "Processando páginas e balões" },
  { id: "context", label: "Contexto", detail: "Buscando sinopse e personagens" },
  { id: "translate", label: "Tradução", detail: "Traduzindo com contexto local" },
  { id: "inpaint", label: "Inpainting", detail: "Removendo texto original" },
  { id: "typeset", label: "Typesetting", detail: "Aplicando texto traduzido" },
] as const;

type JobProcessStepId = (typeof JOB_PROCESS_STEPS)[number]["id"];
type JobProcessStepState = "done" | "active" | "pending";

function JobDetail() {
  const { id } = useParams();
  const query = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["job", id], queryFn: () => api<{ job: Job }>(`/api/jobs/${id}`), enabled: Boolean(id) });
  const events = useLiveEvents(id);
  const cancel = useMutation({ mutationFn: () => api(`/api/jobs/${id}/cancel`, { method: "POST" }), onSuccess: () => query.invalidateQueries({ queryKey: ["job", id] }) });
  const retry = useMutation({ mutationFn: () => api(`/api/jobs/${id}/retry`, { method: "POST" }), onSuccess: () => {
    query.invalidateQueries({ queryKey: ["job", id] });
    query.invalidateQueries({ queryKey: ["jobs"] });
  } });
  const remove = useMutation({ mutationFn: () => api(`/api/jobs/${id}`, { method: "DELETE" }), onSuccess: () => query.invalidateQueries({ queryKey: ["jobs"] }) });
  const timerNow = useJobTimerNow(data?.job);
  if (isLoading || !data?.job) return <section className="desktop-process-screen">Carregando job</section>;
  const job = data.job;
  const processState = buildJobProcessState(job, events, timerNow);
  const logArtifact = job.artifacts?.find((artifact) => artifact.kind === "runner_log" || artifact.kind === "pipeline_log");
  const canCancel = !TERMINAL_JOB_STATUSES.has(job.status);

  return (
    <section className="desktop-process-screen">
      <div className="process-summary">
        <div className="process-heading">
          <h1>{processState.title}</h1>
          <p>{job.obra} - Capítulo {job.capitulo}</p>
        </div>
        <div className="process-progress-meta">
          <strong>{processState.progress}%</strong>
          <span>{processState.etaLabel}</span>
        </div>
        <div className="process-track" aria-label={`Progresso ${processState.progress}%`}>
          <span style={{ width: `${processState.progress}%` }} />
        </div>
        <p className="process-page-label">{processState.pageLabel}</p>
      </div>

      <div className="process-step-list">
        {processState.steps.map((step) => (
          <div className={`process-step process-step-${step.state}`} key={step.id}>
            <span className="process-step-marker">
              {step.state === "done" ? <CheckCircle2 size={16} /> : step.state === "active" ? <span className="process-spinner" /> : <span className="process-dot" />}
            </span>
            <span className="process-step-copy">
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
            {step.state === "active" && step.progress !== null && <span className="process-step-percent">{step.progress}%</span>}
          </div>
        ))}
      </div>

      {job.error_message && <p className="process-error">{job.error_message}</p>}

      <div className="process-actions">
        <button className="process-action process-action-primary" disabled><PauseCircle size={15} />Pausar tradução</button>
        {logArtifact ? (
          <a className="process-action" href={`${API_URL}/api/artifacts/${logArtifact.id}`}><FileText size={15} />Exportar log</a>
        ) : (
          <button className="process-action" disabled><FileText size={15} />Exportar log</button>
        )}
        {canCancel && <button className="process-action" onClick={() => cancel.mutate()} disabled={cancel.isPending}><XCircle size={15} />Cancelar tradução</button>}
        {job.status === "completed" && <Link className="process-action" to={`/resultados/${job.id}`}><Archive size={15} />Resultados</Link>}
        {job.status === "completed" && <Link className="process-action" to={`/projects/${job.id}/preview`}><Eye size={15} />Preview</Link>}
        {(job.status === "failed" || job.status === "cancelled") && <button className="process-action" onClick={() => retry.mutate()} disabled={retry.isPending}><Clock size={15} />Tentar novamente</button>}
        {TERMINAL_JOB_STATUSES.has(job.status) && <button className="process-action process-action-danger" onClick={() => remove.mutate()} disabled={remove.isPending}><Trash2 size={15} />Excluir</button>}
      </div>
      {retry.error && <p className="error">{retry.error.message}</p>}
    </section>
  );
}

function LegacyJobDetail() {
  const { id } = useParams();
  const query = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["job", id], queryFn: () => api<{ job: Job }>(`/api/jobs/${id}`), enabled: Boolean(id) });
  const events = useLiveEvents(id);
  const cancel = useMutation({ mutationFn: () => api(`/api/jobs/${id}/cancel`, { method: "POST" }), onSuccess: () => query.invalidateQueries({ queryKey: ["job", id] }) });
  const retry = useMutation({ mutationFn: () => api(`/api/jobs/${id}/retry`, { method: "POST" }), onSuccess: () => {
    query.invalidateQueries({ queryKey: ["job", id] });
    query.invalidateQueries({ queryKey: ["jobs"] });
  } });
  const remove = useMutation({ mutationFn: () => api(`/api/jobs/${id}`, { method: "DELETE" }), onSuccess: () => query.invalidateQueries({ queryKey: ["jobs"] }) });
  const timerNow = useJobTimerNow(data?.job);
  if (isLoading || !data?.job) return <section className="panel">Carregando job</section>;
  const job = data.job;
  return (
    <section className="panel desktop-job-screen">
      <div className="section-head">
        <div>
          <p className="eyebrow">Job</p>
          <h1>{job.obra}</h1>
          <p>Capítulo {job.capitulo}</p>
        </div>
        <Status value={job.status} />
      </div>
      <div className="metric-grid">
        <Metric label="Modo" value={job.mode} />
        <Metric label="Páginas" value={String(job.page_count ?? "-")} />
        <Metric label="Tempo" value={jobTimeLabel(job, timerNow)} />
        <Metric label="Erro" value={job.error_code ?? "-"} />
      </div>
      {job.error_message && <p className="error">{job.error_message}</p>}
      <div className="action-row">
        <Link className="link-button" to={`/resultados/${job.id}`}><Archive size={16} />Resultados</Link>
        {job.status === "completed" && <Link className="link-button" to={`/projects/${job.id}/preview`}><Eye size={16} />Preview</Link>}
        {(job.status === "failed" || job.status === "cancelled") && <button onClick={() => retry.mutate()} disabled={retry.isPending}><Clock size={16} />Tentar novamente</button>}
        <button onClick={() => cancel.mutate()} disabled={cancel.isPending}><XCircle size={16} />Cancelar</button>
        <button className="danger" onClick={() => remove.mutate()} disabled={remove.isPending}><Trash2 size={16} />Excluir</button>
      </div>
      {retry.error && <p className="error">{retry.error.message}</p>}
      <h2>Eventos</h2>
      <div className="event-list">
        {events.length ? events.map((event, index) => (
          <div className="event-item" key={`${event.created_at}-${index}`}>
            <span>{event.stage}</span>
            <strong>{event.message}</strong>
          </div>
        )) : <p className="empty">Aguardando eventos do worker.</p>}
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><Gauge size={15} /><span>{label}</span><strong>{value}</strong></div>;
}

function Results() {
  const { id } = useParams();
  const { data } = useQuery({ queryKey: ["job", id], queryFn: () => api<{ job: Job }>(`/api/jobs/${id}`), enabled: Boolean(id) });
  const artifacts = data?.job.artifacts ?? [];
  return (
    <section className="panel desktop-results-screen">
      <div className="section-head">
        <div>
          <p className="eyebrow">Artefatos</p>
          <h1>Resultados</h1>
        </div>
        <div className="action-row">
          {id && <Link className="link-button" to={`/projects/${id}/preview`}><Eye size={16} />Preview</Link>}
          {id && <a className="primary link-button" href={`${API_URL}/api/jobs/${id}/download/zip`}><Download size={16} />Baixar ZIP</a>}
        </div>
      </div>
      <div className="artifact-list">
        {artifacts.map((artifact) => (
          <a key={artifact.id} href={`${API_URL}/api/artifacts/${artifact.id}`} className="artifact-row">
            <span><FileText size={14} />{artifactLabel(artifact.kind)}</span>
            <strong>{artifact.filename}</strong>
            <small>{formatBytes(artifact.size)}</small>
          </a>
        ))}
      </div>
    </section>
  );
}

function SettingsPage() {
  const apiUrl = useMemo(() => API_URL, []);
  return (
    <section className="panel narrow desktop-settings-screen">
      <p className="eyebrow">Local</p>
      <h1>Config</h1>
      <Metric label="API" value={apiUrl} />
      <Metric label="Worker" value="admin-pc" />
      <Metric label="Cobrança" value="desligada no beta" />
    </section>
  );
}

function artifactLabel(kind: string) {
  const labels: Record<string, string> = {
    input_original: "Entrada",
    input_archive: "Arquivo original",
    translated_image: "Imagem traduzida",
    project_json: "Projeto",
    pipeline_log: "Pipeline log",
    runner_log: "Worker log",
    bundle_zip: "ZIP",
  };
  return labels[kind] ?? kind;
}

function formatBytes(size: number) {
  if (size < 1024 * 1024) return `${Math.ceil(size / 1024)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function parseJobTime(value?: string) {
  if (!value) return null;
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const timestamp = new Date(/(?:Z|[+-]\d\d:?\d\d)$/.test(normalized) ? normalized : `${normalized}Z`).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
}

function formatDuration(totalSeconds: number) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  const paddedMinutes = String(minutes).padStart(2, "0");
  const paddedSeconds = String(remainingSeconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${paddedMinutes}:${paddedSeconds}` : `${paddedMinutes}:${paddedSeconds}`;
}

function jobTimeLabel(job: Job, now: number) {
  const hasFinalDuration = typeof job.processing_seconds === "number" && Number.isFinite(job.processing_seconds);
  const startedAt = parseJobTime(job.started_at ?? job.created_at);
  const isTerminal = TERMINAL_JOB_STATUSES.has(job.status);
  if (!isTerminal && startedAt !== null) {
    const liveSeconds = (now - startedAt) / 1000;
    return formatDuration(hasFinalDuration ? Math.max(liveSeconds, job.processing_seconds ?? 0) : liveSeconds);
  }
  if (hasFinalDuration) return formatDuration(job.processing_seconds ?? 0);
  if (startedAt === null) return "-";
  const finishedAt = parseJobTime(job.finished_at);
  if (finishedAt !== null) return formatDuration((finishedAt - startedAt) / 1000);
  if (isTerminal) return "-";
  return formatDuration((now - startedAt) / 1000);
}

function buildJobProcessState(job: Job, events: JobEvent[], now: number) {
  const latestProgressEvent = events.find((event) => hasProgressPayload(event.payload));
  const latestEvent = latestProgressEvent ?? events[0];
  const payload = latestProgressEvent?.payload ?? {};
  const explicitProgress = readPayloadNumber(payload, "overall_progress");
  const progress = job.status === "completed" ? 100 : clampPercent(explicitProgress ?? fallbackJobProgress(job.status));
  const activeStage = resolveActiveProcessStage(job, latestEvent, progress);
  const activeIndex = JOB_PROCESS_STEPS.findIndex((step) => step.id === activeStage);
  const stepProgress = activeStage ? clampPercent(readPayloadNumber(payload, "step_progress") ?? progress) : null;
  const etaSeconds = readPayloadNumber(payload, "eta_seconds") ?? estimateEtaSeconds(job, progress, now);
  const pageLabel = buildPageLabel(job, payload);

  return {
    title: processTitle(job.status),
    progress,
    etaLabel: etaSeconds !== null && !TERMINAL_JOB_STATUSES.has(job.status) ? `~${formatCompactDuration(etaSeconds)} restante` : processTimeFallback(job, now),
    pageLabel,
    steps: JOB_PROCESS_STEPS.map((step, index) => {
      const state: JobProcessStepState = job.status === "completed" || (activeIndex >= 0 && index < activeIndex) ? "done" : index === activeIndex ? "active" : "pending";
      const activeMessage = state === "active" && latestEvent?.message ? latestEvent.message : "";
      return {
        ...step,
        state,
        detail: activeMessage || step.detail,
        progress: state === "active" ? stepProgress : null,
      };
    }),
  };
}

function hasProgressPayload(payload?: Record<string, unknown>) {
  return readPayloadNumber(payload, "overall_progress") !== null || readPayloadNumber(payload, "step_progress") !== null || readPayloadText(payload, "type") === "progress";
}

function readPayloadNumber(payload: Record<string, unknown> | undefined, key: string) {
  const value = payload?.[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function readPayloadText(payload: Record<string, unknown> | undefined, key: string) {
  const value = payload?.[key];
  return typeof value === "string" ? value : null;
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function fallbackJobProgress(status: string) {
  if (status === "queued") return 4;
  if (status === "claimed") return 8;
  if (status === "running") return 12;
  if (status === "uploading_results") return 96;
  if (status === "completed") return 100;
  return 0;
}

function resolveActiveProcessStage(job: Job, event: JobEvent | undefined, progress: number): JobProcessStepId | null {
  if (job.status === "completed") return "typeset";
  if (job.status === "queued" || job.status === "claimed") return "extract";
  if (job.status === "uploading_results") return "typeset";
  const payloadStage = readPayloadText(event?.payload, "step") ?? readPayloadText(event?.payload, "stage");
  const mapped = mapProcessStage(payloadStage ?? event?.stage ?? event?.message ?? "");
  if (mapped) return mapped;
  if (progress < 10) return "extract";
  if (progress < 70) return "ocr";
  if (progress < 78) return "context";
  if (progress < 86) return "translate";
  if (progress < 94) return "inpaint";
  return "typeset";
}

function mapProcessStage(value: string): JobProcessStepId | null {
  const normalized = value.toLocaleLowerCase("pt-BR");
  if (!normalized) return null;
  if (normalized.includes("extract") || normalized.includes("queue") || normalized.includes("worker") || normalized.includes("arquivo")) return "extract";
  if (normalized.includes("ocr") || normalized.includes("detect") || normalized.includes("recognize") || normalized.includes("texto")) return "ocr";
  if (normalized.includes("context") || normalized.includes("gloss") || normalized.includes("sinopse") || normalized.includes("personagem")) return "context";
  if (normalized.includes("translat") || normalized.includes("traduz")) return "translate";
  if (normalized.includes("inpaint") || normalized.includes("limp") || normalized.includes("remove")) return "inpaint";
  if (normalized.includes("typeset") || normalized.includes("render") || normalized.includes("artifact") || normalized.includes("done") || normalized.includes("final")) return "typeset";
  return null;
}

function buildPageLabel(job: Job, payload: Record<string, unknown>) {
  const currentPage = readPayloadNumber(payload, "current_page") ?? 0;
  const totalPages = readPayloadNumber(payload, "total_pages") ?? job.page_count ?? 0;
  return `Página ${Math.max(0, Math.floor(currentPage))}/${Math.max(0, Math.floor(totalPages))}`;
}

function estimateEtaSeconds(job: Job, progress: number, now: number) {
  const startedAt = parseJobTime(job.started_at ?? job.created_at);
  if (startedAt === null || progress <= 4 || progress >= 100) return null;
  const elapsed = Math.max(0, (now - startedAt) / 1000);
  if (elapsed < 5) return null;
  return (elapsed * (100 - progress)) / progress;
}

function processTitle(status: string) {
  if (status === "completed") return "Tradução concluída";
  if (status === "failed") return "Tradução falhou";
  if (status === "cancelled") return "Tradução cancelada";
  if (status === "queued" || status === "claimed") return "Preparando...";
  return "Traduzindo...";
}

function processTimeFallback(job: Job, now: number) {
  if (job.status === "completed") return `tempo total ${jobTimeLabel(job, now)}`;
  if (job.status === "failed" || job.status === "cancelled") return jobTimeLabel(job, now);
  return "calculando...";
}

function formatCompactDuration(totalSeconds: number) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${remainingSeconds}s`;
  return `${remainingSeconds}s`;
}

function useJobTimerNow(job?: Job) {
  const shouldTick = Boolean(job && !TERMINAL_JOB_STATUSES.has(job.status));
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!shouldTick) {
      setNow(Date.now());
      return;
    }
    setNow(Date.now());
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [shouldTick, job?.id, job?.created_at, job?.started_at, job?.finished_at, job?.processing_seconds]);
  return now;
}

function Legal() {
  return (
    <section className="panel narrow legal-text">
      <p className="eyebrow">Legal</p>
      <h1>Privacidade</h1>
      <p>O beta processa arquivos no worker local. A API registra jobs, artifacts, eventos e uso para validar o fluxo SaaS-ready.</p>
      <p>O TraduzAI não hospeda nem fornece obras protegidas. O usuário é responsável pelos arquivos usados.</p>
    </section>
  );
}

function Admin() {
  const { data, isLoading } = useQuery({ queryKey: ["admin-overview"], queryFn: () => api<AdminOverview>("/api/admin/overview") });
  return (
    <section className="panel desktop-admin-screen">
      <p className="eyebrow">Admin</p>
      <h1>Operação</h1>
      {isLoading && <p>Carregando operação</p>}
      <div className="metric-grid">
        <Metric label="Org" value="default" />
        <Metric label="Worker auth" value="Bearer token" />
        <Metric label="Concorrência" value="por worker" />
        <Metric label="Workers" value={String(data?.workers.length ?? 0)} />
      </div>
      <h2>Workers</h2>
      <div className="event-list">
        {(data?.workers ?? []).map((worker) => (
          <div className="event-item" key={worker.id}>
            <span>{worker.status}</span>
            <strong>{worker.name}</strong>
            <small>max {worker.max_concurrent_jobs}</small>
          </div>
        ))}
      </div>
      <h2>Jobs recentes</h2>
      <JobTable jobs={data?.jobs ?? []} />
      <h2>Audit</h2>
      <div className="event-list">
        {(data?.audit_logs ?? []).map((item) => (
          <div className="event-item" key={item.id}>
            <span>{item.entity_type}</span>
            <strong>{item.action}</strong>
            <small>{item.entity_id.slice(0, 8)}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/" element={<Landing />} />
      <Route path="/dashboard" element={<Protected><DashboardEntry /></Protected>} />
      <Route path="/novo" element={<Navigate to="/projects/new" replace />} />
      <Route path="/projects" element={<Protected><ProjectsPage /></Protected>} />
      <Route path="/projects/new" element={<Protected><ProjectSetup /></Protected>} />
      <Route path="/job/:id" element={<Protected><JobDetail /></Protected>} />
      <Route path="/resultados/:id" element={<Protected><Results /></Protected>} />
      <Route path="/projects/:id/preview" element={<Protected><ProjectPreview /></Protected>} />
      <Route path="/projects/:id/editor" element={<ProtectedFullScreen><WebEditorRoute /></ProtectedFullScreen>} />
      <Route path="/projects/:id/settings" element={<Protected><ProjectSettings /></Protected>} />
      <Route path="/settings" element={<Protected><SettingsPage /></Protected>} />
      <Route path="/legal" element={<Legal />} />
      <Route path="/admin" element={<Protected><Admin /></Protected>} />
    </Routes>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
