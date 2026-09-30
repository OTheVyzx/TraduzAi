import { ArrowDownAZ, Check, RotateCcw, X } from "lucide-react";
import { useEffect, useState } from "react";
import type { SourceFilter, SourceFilterChange } from "./sourceRuntimeClient";

type EditableSourceFilter = Extract<SourceFilter, { value: unknown }>;

function editable(filters: SourceFilter[]): EditableSourceFilter[] {
  return filters.flatMap((filter): EditableSourceFilter[] => filter.type === "group" ? editable(filter.children) : filter.type === "header" || filter.type === "separator" ? [] : [filter]);
}

function initialValues(filters: SourceFilter[]): Record<string, unknown> {
  return Object.fromEntries(editable(filters).map((filter) => [filter.path.join("."), filter.value]));
}

export function ReaderSourceFilters({ open, filters, busy, error, onClose, onApply }: {
  open: boolean;
  filters: SourceFilter[];
  busy: boolean;
  error?: string | null;
  onClose(): void;
  onApply(changes: SourceFilterChange[]): void;
}) {
  const [values, setValues] = useState<Record<string, unknown>>({});
  useEffect(() => { if (open) setValues(initialValues(filters)); }, [open, filters]);
  if (!open) return null;
  const set = (filter: SourceFilter, value: unknown) => setValues((current) => ({ ...current, [filter.path.join(".")]: value }));
  const controls = (items: SourceFilter[]) => items.map((filter) => {
    if (filter.type === "header") return <h3 key={filter.path.join(".")}>{filter.name}</h3>;
    if (filter.type === "separator") return <hr key={filter.path.join(".")} />;
    if (filter.type === "group") return <fieldset key={filter.path.join(".")}><legend>{filter.name}</legend>{controls(filter.children)}</fieldset>;
    const key = filter.path.join(".");
    if (filter.type === "select") return <label key={key}><span>{filter.name}</span><select value={Number(values[key] ?? filter.value)} onChange={(event) => set(filter, Number(event.currentTarget.value))}>{filter.values.map((value, index) => <option value={index} key={value}>{value}</option>)}</select></label>;
    if (filter.type === "text") return <label key={key}><span>{filter.name}</span><input value={String(values[key] ?? filter.value)} onChange={(event) => set(filter, event.currentTarget.value)} /></label>;
    if (filter.type === "checkbox") return <label className="reader-filter-check" key={key}><input type="checkbox" checked={Boolean(values[key] ?? filter.value)} onChange={(event) => set(filter, event.currentTarget.checked)} /><span>{filter.name}</span></label>;
    if (filter.type === "tristate") return <label key={key}><span>{filter.name}</span><select value={Number(values[key] ?? filter.value)} onChange={(event) => set(filter, Number(event.currentTarget.value))}><option value={0}>Ignorar</option><option value={1}>Incluir</option><option value={2}>Excluir</option></select></label>;
    const sort = (values[key] ?? filter.value) as { index: number; ascending: boolean } | null;
    return <div className="reader-filter-sort" key={key}><label><span>{filter.name}</span><select value={sort?.index ?? -1} onChange={(event) => set(filter, event.currentTarget.value === "-1" ? null : { index: Number(event.currentTarget.value), ascending: sort?.ascending ?? true })}><option value={-1}>Padrão</option>{filter.values.map((value, index) => <option value={index} key={value}>{value}</option>)}</select></label>{sort && <button type="button" aria-label="Alternar direção" onClick={() => set(filter, { ...sort, ascending: !sort.ascending })}><ArrowDownAZ size={17} className={sort.ascending ? "" : "reader-filter-desc"} /></button>}</div>;
  });
  return <div className="studio-floating-layer reader-filter-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside className="reader-filter-panel" role="dialog" aria-modal="true" aria-label="Filtros da fonte">
      <header><div><small>Filtros da extensão</small><h2>Refinar catálogo</h2></div><button type="button" aria-label="Fechar filtros" onClick={onClose}><X size={20} /></button></header>
      <div className="reader-filter-fields">{busy ? <p>Carregando filtros…</p> : error ? <p className="reader-filter-error">{error}</p> : filters.length ? controls(filters) : <p>Esta fonte não oferece filtros.</p>}</div>
      <footer><button type="button" onClick={() => setValues(initialValues(filters))}><RotateCcw size={15} /> Limpar</button><button type="button" disabled={busy || !editable(filters).length} onClick={() => onApply(editable(filters).map((filter) => ({ path: filter.path, value: values[filter.path.join(".")] ?? filter.value })))}><Check size={15} /> Aplicar</button></footer>
    </aside>
  </div>;
}
