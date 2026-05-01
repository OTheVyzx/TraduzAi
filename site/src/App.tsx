import { faq, features, plans, steps } from "./content";

const asset = (name: string) => `/assets/${name}.png`;

function Nav() {
  return (
    <header className="site-nav">
      <a className="brand" href="/">
        TraduzAI
      </a>
      <nav>
        <a href="/download">Download</a>
        <a href="/docs">Docs</a>
        <a href="/legal">Legal</a>
        <a href="/roadmap">Roadmap</a>
      </nav>
    </header>
  );
}

function Hero() {
  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">Local-first para quadrinhos</p>
        <h1>TraduzAI</h1>
        <p className="lead">
          Traduza mangas, manhwas e manhuas com IA localmente, com contexto, glossario e revisao visual.
        </p>
        <p>
          Importe um capitulo, selecione a obra, deixe a IA detectar, traduzir, limpar baloes e recriar o texto.
          Revise tudo antes de exportar.
        </p>
        <div className="actions">
          <a className="button primary" href="/download">
            Baixar para Windows
          </a>
          <a className="button ghost" href="#demo">
            Ver demonstracao
          </a>
        </div>
      </div>
      <img className="hero-image" src={asset("hero-app-mockup")} alt="Mockup sintetico do app TraduzAI" />
    </section>
  );
}

function Landing() {
  return (
    <>
      <Hero />
      <section id="demo" className="section visual-grid">
        <div>
          <p className="eyebrow">Demonstracao visual</p>
          <h2>Antes, contexto, revisao e export em um fluxo unico.</h2>
        </div>
        <img src={asset("before-after-demo")} alt="Demo sintetica de antes e depois" />
      </section>

      <section className="section">
        <h2>Como funciona</h2>
        <div className="steps">
          {steps.map((step, index) => (
            <article className="step-card" key={step}>
              <span>{index + 1}</span>
              <p>{step}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>Diferenciais</h2>
        <div className="cards">
          {features.map((feature) => (
            <article className="card" key={feature.title}>
              <h3>{feature.title}</h3>
              <p>{feature.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section product-panels">
        <img src={asset("context-panel")} alt="Painel sintetico de contexto" />
        <img src={asset("glossary-panel")} alt="Painel sintetico de glossario" />
        <img src={asset("qa-report")} alt="Relatorio sintetico de QA" />
        <img src={asset("editor-view")} alt="Editor sintetico do TraduzAI" />
      </section>

      <section className="section pricing">
        <h2>Planos</h2>
        <div className="cards">
          {plans.map((plan) => (
            <article className="card price-card" key={plan.name}>
              <h3>{plan.name}</h3>
              <p className="price">{plan.price}</p>
              <ul>
                {plan.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="section faq">
        <h2>FAQ</h2>
        {faq.map((item) => (
          <details key={item.question}>
            <summary>{item.question}</summary>
            <p>{item.answer}</p>
          </details>
        ))}
      </section>

      <section className="section legal-callout">
        <h2>Aviso legal</h2>
        <p>
          O TraduzAI e uma ferramenta local de edicao, OCR, traducao e typesetting. Nao hospeda, distribui ou
          fornece obras protegidas por direitos autorais. O usuario e responsavel pelos arquivos utilizados.
        </p>
        <a className="button ghost" href="/legal">
          Ler privacidade e termos
        </a>
      </section>
    </>
  );
}

function DownloadPage() {
  return (
    <main className="page">
      <h1>Download</h1>
      <div className="download-card">
        <img src={asset("export-options")} alt="Opcoes sinteticas de export" />
        <div>
          <h2>Windows beta</h2>
          <p>Versao v0.2.0-beta para testes locais. macOS e Linux entram no roadmap publico.</p>
          <a className="button primary" href="#waitlist">
            Entrar na lista de espera
          </a>
          <p className="muted">Checksums e changelog serao publicados junto do instalador beta.</p>
        </div>
      </div>
    </main>
  );
}

function DocsPage() {
  return (
    <main className="page">
      <h1>Docs</h1>
      <div className="cards">
        {["Como instalar", "Como importar capitulo", "Como buscar contexto", "Como revisar glossario", "Como exportar", "FAQ"].map(
          (item) => (
            <article className="card" key={item}>
              <h2>{item}</h2>
              <p>Guia pratico disponivel na documentacao do repositorio.</p>
            </article>
          ),
        )}
      </div>
    </main>
  );
}

function LegalPage() {
  return (
    <main className="page">
      <h1>Legal e privacidade</h1>
      <section className="card legal-text">
        <h2>Aviso legal</h2>
        <p>O TraduzAI nao hospeda, distribui ou fornece obras protegidas por direitos autorais.</p>
        <p>O usuario e responsavel pelos arquivos utilizados e pelo direito de processa-los.</p>
      </section>
      <section className="card legal-text">
        <h2>Privacidade</h2>
        <p>Os arquivos ficam no computador do usuario.</p>
        <p>O app so acessa a internet para traducao e busca de contexto, quando ativado.</p>
        <p>Nenhuma pagina e enviada para fontes de contexto.</p>
      </section>
    </main>
  );
}

function RoadmapPage() {
  return (
    <main className="page">
      <h1>Roadmap publico</h1>
      <div className="steps">
        {["Beta Windows", "Instalador assinado", "Export avancado", "Modo lote", "macOS e Linux"].map((item, index) => (
          <article className="step-card" key={item}>
            <span>{index + 1}</span>
            <p>{item}</p>
          </article>
        ))}
      </div>
    </main>
  );
}

function Router() {
  const path = window.location.pathname;
  if (path === "/download") return <DownloadPage />;
  if (path === "/docs") return <DocsPage />;
  if (path === "/legal") return <LegalPage />;
  if (path === "/roadmap") return <RoadmapPage />;
  return <Landing />;
}

export function App() {
  return (
    <div className="min-h-screen">
      <Nav />
      <Router />
      <footer>
        <span>TraduzAI v0.2.0-beta</span>
        <span>Ferramenta local. Sem distribuicao de obras.</span>
      </footer>
    </div>
  );
}
