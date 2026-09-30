import { useMemo, useState } from "react";
import { BookOpen, Check, Languages, LayoutPanelTop, RotateCcw, Save, SlidersHorizontal } from "lucide-react";
import type { StudioLibrary } from "../library/libraryModel";
import { createStudioSettingsDraft, isStudioSettingsDirty, resetStudioSettings, type StudioSettingsDraft } from "./studioSettings";

export function StudioSettingsView({
  preferences,
  saving,
  onSave,
}: {
  preferences: StudioLibrary["preferences"];
  saving: boolean;
  onSave: (draft: StudioSettingsDraft) => Promise<void>;
}) {
  const saved = useMemo(() => createStudioSettingsDraft(preferences), [preferences]);
  const [draft, setDraft] = useState(saved);
  const dirty = isStudioSettingsDirty(draft, saved);
  const save = async () => { if (dirty) await onSave(draft); };
  const restore = () => {
    if (window.confirm("Restaurar as preferências do Studio?")) setDraft(resetStudioSettings());
  };
  return <section className="studio-settings-view">
    <aside className="studio-settings-sidebar"><h1>Configurações</h1><button type="button" aria-current="page"><SlidersHorizontal /> Geral <small>Preferências ativas</small></button><button type="button"><BookOpen /> Biblioteca <small>Visualização e miniaturas</small></button><button type="button"><Languages /> Tradução <small>Idioma de tracking</small></button></aside>
    <main className="studio-settings-main"><header><div><h1>Configurações</h1><p>Personalize o TraduzAI Studio de acordo com o seu fluxo de trabalho.</p></div><div><button type="button" onClick={restore}><RotateCcw /> Restaurar padrão</button><button type="button" disabled={!dirty || saving} onClick={() => void save()}><Save /> {saving ? "Salvando…" : "Salvar alterações"}</button></div></header>
      <div className="studio-settings-cards">
        <section><h2><LayoutPanelTop /> Biblioteca</h2><label><span>Visualização padrão dos capítulos</span><select value={draft.defaultChapterView} onChange={(event) => setDraft({ ...draft, defaultChapterView: event.currentTarget.value as StudioSettingsDraft["defaultChapterView"] })}><option value="list">Lista</option><option value="grid">Grade</option></select></label><label><span>Tamanho padrão das miniaturas</span><input type="range" min="112" max="240" step="8" value={draft.thumbnailSize} onChange={(event) => setDraft({ ...draft, thumbnailSize: Number(event.currentTarget.value) })} /><b>{draft.thumbnailSize}px</b></label></section>
        <section><h2><Languages /> Tracking</h2><label><span>Idioma da fonte de atualizações</span><select value={draft.trackingLanguage} onChange={(event) => setDraft({ ...draft, trackingLanguage: event.currentTarget.value })}><option value="en">Inglês</option><option value="pt-BR">Português (Brasil)</option><option value="ko">Coreano</option></select></label><p className="studio-settings-note"><Check /> Esta preferência é salva no catálogo local do Studio.</p></section>
        <section className="studio-settings-unavailable"><h2>OCR e Tradução</h2><p>Os motores, modelos e credenciais são definidos pelo runtime do pipeline. Esta tela não mostra opções fictícias.</p><span>Indisponível até uma consulta real de capacidades.</span></section>
        <section className="studio-settings-unavailable"><h2>Exportação e armazenamento</h2><p>Formatos disponíveis e limpeza de cache dependem do backend Tauri e do projeto aberto.</p><span>Indisponível até uma consulta real de capacidades.</span></section>
      </div>
    </main>
    <aside className="studio-settings-summary"><h2>Resumo do sistema</h2><p>Dados de máquina e serviços serão exibidos somente quando obtidos do runtime.</p><div><strong>Catálogo do Studio</strong><span>{saving ? "Salvando" : "Local"}</span></div><div><strong>Pipeline</strong><span>Não consultado</span></div></aside>
  </section>;
}
