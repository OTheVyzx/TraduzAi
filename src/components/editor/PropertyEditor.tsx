import { useEditorStore } from "../../lib/stores/editorStore";
import {
  AlignLeft,
  AlignCenter,
  AlignRight,
  Bold,
  Italic,
  ChevronDown,
  ChevronRight,
  Search,
  RefreshCw,
  Link2Off,
} from "lucide-react";
import { useState } from "react";

const AVAILABLE_FONTS = [
  "ComicNeue-Bold.ttf",
  "Newrotic.ttf",
  "CCDaveGibbonsLower W00 Regular.ttf",
  "KOMIKAX_.ttf",
];

function Section({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-white/5">
      <button
        className="flex items-center gap-1.5 w-full px-4 py-2 text-xs font-medium text-text-secondary hover:text-text-primary transition-smooth"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </button>
      {open && <div className="px-4 pb-3 space-y-2">{children}</div>}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-xs text-text-muted mb-0.5">{children}</label>;
}

function InputField({
  value,
  onChange,
  type = "text",
  readOnly = false,
  title = "Campo de entrada",
}: {
  value: string | number;
  onChange: (val: string) => void;
  type?: "text" | "number";
  readOnly?: boolean;
  title?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      title={title}
      onChange={(e) => onChange(e.target.value)}
      readOnly={readOnly}
      className={`w-full px-2 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm text-text-primary
        focus:border-accent-purple/50 focus:outline-none transition-smooth
        ${readOnly ? "opacity-60 cursor-default" : ""}`}
    />
  );
}

export function PropertyEditor() {
  const { 
    selectedLayerId, 
    pendingEdits, 
    currentPage, 
    isRetypesetting 
  } = useEditorStore();
  
  const updateEdit = useEditorStore((s) => s.updatePendingEdit);
  const updateEstilo = useEditorStore((s) => s.updatePendingEstilo);
  
  const selectedLayer = currentPage?.text_layers.find((t) => t.id === selectedLayerId);
  const entry = selectedLayer;

  if (!entry || !selectedLayerId) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <p className="text-xs text-text-muted text-center">
          Selecione uma camada para editar
        </p>
      </div>
    );
  }

  const edit = pendingEdits[selectedLayerId];
  const traduzido = edit?.traduzido ?? edit?.translated ?? entry.traduzido ?? entry.translated ?? "";
  const original = entry.original;
  const estilo = edit?.estilo ? { ...entry.estilo, ...edit.estilo } : entry.estilo;
  const [x1, y1, x2, y2] = edit?.bbox ?? entry.layout_bbox ?? entry.bbox;

  const confidencePercent = Math.round(((entry.confianca_ocr ?? entry.ocr_confidence ?? 0) || 0) * 100);
  const confidenceColor =
    confidencePercent >= 80
      ? "text-status-success"
      : confidencePercent >= 50
        ? "text-status-warning"
        : "text-status-error";

  return (
    <div className="flex flex-col h-full bg-bg-secondary/10">
      <div className="flex-1 overflow-y-auto">
      <div className="border-b border-white/5 bg-bg-primary/35 px-4 py-3">
        <div className="flex items-center justify-between text-[11px] text-text-muted">
          <span className="rounded-full border border-white/10 px-2 py-1 uppercase tracking-[0.14em]">
            {entry.tipo}
          </span>
          <span className={`font-mono ${confidenceColor}`}>{confidencePercent}% OCR</span>
        </div>
        <p className="mt-2 text-xs text-text-secondary">
          Caixa atual: {x1}, {y1}, {x2}, {y2}
        </p>
      </div>
      {/* Texto */}
      <Section title="Texto" defaultOpen={true}>
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <Label>Texto Original</Label>
            <span className={`text-[10px] font-mono ${confidenceColor}`}>
              {confidencePercent}%
            </span>
          </div>
          <textarea
            value={original}
            title="Texto original (somente leitura)"
            readOnly
            rows={3}
            className="w-full px-2 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm
              text-text-secondary resize-none opacity-60 cursor-default"
          />
        </div>
        <div>
          <Label>Tradução</Label>
          <textarea
            value={traduzido}
            title="Texto traduzido"
            onChange={(e) =>
              updateEdit(selectedLayerId, {
                traduzido: e.target.value,
                translated: e.target.value,
              })
            }
            rows={5}
            className="w-full px-2 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm
              text-text-primary resize-y focus:border-accent-purple/50 focus:outline-none transition-smooth"
          />
          <p className="mt-1 text-[11px] text-text-muted">
            Ctrl+S salva as alteracoes; o overlay no canvas atualiza em tempo real.
          </p>
        </div>
      </Section>

      {/* Posição */}
      <Section title="Posição" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label>X</Label>
            <InputField
              type="number"
              value={x1}
              onChange={(v) => {
                const nx = parseInt(v) || 0;
                const w = x2 - x1;
                updateEdit(selectedLayerId, { bbox: [nx, y1, nx + w, y2] });
              }}
            />
          </div>
          <div>
            <Label>Y</Label>
            <InputField
              type="number"
              value={y1}
              onChange={(v) => {
                const ny = parseInt(v) || 0;
                const h = y2 - y1;
                updateEdit(selectedLayerId, { bbox: [x1, ny, x2, ny + h] });
              }}
            />
          </div>
          <div>
            <Label>Largura</Label>
            <InputField
              type="number"
              value={x2 - x1}
              onChange={(v) => {
                const w = parseInt(v) || 1;
                updateEdit(selectedLayerId, { bbox: [x1, y1, x1 + w, y2] });
              }}
            />
          </div>
          <div>
            <Label>Altura</Label>
            <InputField
              type="number"
              value={y2 - y1}
              onChange={(v) => {
                const h = parseInt(v) || 1;
                updateEdit(selectedLayerId, { bbox: [x1, y1, x2, y1 + h] });
              }}
            />
          </div>
        </div>
      </Section>

      {/* Estilo */}
      <Section title="Estilo" defaultOpen={false}>
        <div>
          <Label>Fonte</Label>
          <select
            value={estilo.fonte}
            title="Selecionar fonte"
            onChange={(e) => updateEstilo(selectedLayerId, { fonte: e.target.value })}
            className="w-full px-2 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm
              text-text-primary focus:border-accent-purple/50 focus:outline-none transition-smooth"
          >
            {AVAILABLE_FONTS.map((f) => (
              <option key={f} value={f}>
                {f.replace(/\.(ttf|otf)$/i, "")}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label>Tamanho</Label>
            <InputField
              type="number"
              value={estilo.tamanho}
              onChange={(v) => updateEstilo(selectedLayerId, { tamanho: parseInt(v) || 12 })}
            />
          </div>
          <div>
            <Label>Cor</Label>
            <div className="flex items-center gap-1.5">
              <input
                type="color"
                value={estilo.cor}
                title="Cor do texto"
                onChange={(e) => updateEstilo(selectedLayerId, { cor: e.target.value })}
                className="w-7 h-7 rounded border border-white/10 cursor-pointer bg-transparent"
              />
              <InputField
                value={estilo.cor}
                onChange={(v) => updateEstilo(selectedLayerId, { cor: v })}
              />
            </div>
          </div>
        </div>

        {/* Alignment */}
        <div>
          <Label>Alinhamento</Label>
          <div className="flex gap-1">
            {(["left", "center", "right"] as const).map((align) => {
              const Icon = align === "left" ? AlignLeft : align === "center" ? AlignCenter : AlignRight;
              return (
                <button
                  key={align}
                  title={`Alinhamento ${align}`}
                  onClick={() => updateEstilo(selectedLayerId, { alinhamento: align })}
                  className={`p-1.5 rounded transition-smooth ${
                    estilo.alinhamento === align
                      ? "bg-accent-purple/20 text-accent-purple"
                      : "bg-bg-tertiary text-text-secondary hover:text-text-primary"
                  }`}
                >
                  <Icon size={14} />
                </button>
              );
            })}

            <div className="w-px bg-white/5 mx-1" />

            <button
              onClick={() => updateEstilo(selectedLayerId, { bold: !estilo.bold })}
              title="Negrito"
              className={`p-1.5 rounded transition-smooth ${
                estilo.bold
                  ? "bg-accent-purple/20 text-accent-purple"
                  : "bg-bg-tertiary text-text-secondary hover:text-text-primary"
              }`}
            >
              <Bold size={14} />
            </button>
            <button
              onClick={() => updateEstilo(selectedLayerId, { italico: !estilo.italico })}
              title="Itálico"
              className={`p-1.5 rounded transition-smooth ${
                estilo.italico
                  ? "bg-accent-purple/20 text-accent-purple"
                  : "bg-bg-tertiary text-text-secondary hover:text-text-primary"
              }`}
            >
              <Italic size={14} />
            </button>
          </div>
        </div>
      </Section>

      {/* Efeitos */}
      <Section title="Efeitos" defaultOpen={false}>
        {/* Contorno */}
        <div>
          <Label>Contorno</Label>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={estilo.contorno || "#000000"}
              title="Cor do contorno"
              onChange={(e) => updateEstilo(selectedLayerId, { contorno: e.target.value })}
              className="w-7 h-7 rounded border border-white/10 cursor-pointer bg-transparent"
            />
            <div className="flex-1">
              <InputField
                type="number"
                value={estilo.contorno_px}
                onChange={(v) => updateEstilo(selectedLayerId, { contorno_px: parseInt(v) || 0 })}
              />
            </div>
            <span className="text-xs text-text-muted">px</span>
          </div>
        </div>

        {/* Glow */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Label>Brilho</Label>
            <button
              onClick={() => updateEstilo(selectedLayerId, { glow: !estilo.glow })}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-smooth ${
                estilo.glow
                  ? "bg-accent-cyan/20 text-accent-cyan"
                  : "bg-bg-tertiary text-text-muted"
              }`}
            >
              {estilo.glow ? "ON" : "OFF"}
            </button>
          </div>
          {estilo.glow && (
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={estilo.glow_cor || "#FFFFFF"}
                title="Cor do brilho"
                onChange={(e) => updateEstilo(selectedLayerId, { glow_cor: e.target.value })}
                className="w-7 h-7 rounded border border-white/10 cursor-pointer bg-transparent"
              />
              <div className="flex-1">
                <InputField
                  type="number"
                  value={estilo.glow_px}
                  onChange={(v) => updateEstilo(selectedLayerId, { glow_px: parseInt(v) || 0 })}
                />
              </div>
              <span className="text-xs text-text-muted">px</span>
            </div>
          )}
        </div>

        {/* Sombra */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Label>Sombra</Label>
            <button
              onClick={() => updateEstilo(selectedLayerId, { sombra: !estilo.sombra })}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-smooth ${
                estilo.sombra
                  ? "bg-accent-cyan/20 text-accent-cyan"
                  : "bg-bg-tertiary text-text-muted"
              }`}
            >
              {estilo.sombra ? "ON" : "OFF"}
            </button>
          </div>
          {estilo.sombra && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={estilo.sombra_cor || "#000000"}
                  title="Cor da sombra"
                  onChange={(e) => updateEstilo(selectedLayerId, { sombra_cor: e.target.value })}
                  className="w-7 h-7 rounded border border-white/10 cursor-pointer bg-transparent"
                />
                <span className="text-xs text-text-muted">cor</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label>Offset X</Label>
                  <InputField
                    type="number"
                    value={estilo.sombra_offset?.[0] ?? 0}
                    onChange={(v) =>
                      updateEstilo(selectedLayerId, {
                        sombra_offset: [parseInt(v) || 0, estilo.sombra_offset?.[1] ?? 0],
                      })
                    }
                  />
                </div>
                <div>
                  <Label>Offset Y</Label>
                  <InputField
                    type="number"
                    value={estilo.sombra_offset?.[1] ?? 0}
                    onChange={(v) =>
                      updateEstilo(selectedLayerId, {
                        sombra_offset: [estilo.sombra_offset?.[0] ?? 0, parseInt(v) || 0],
                      })
                    }
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </Section>
    </div>

    {/* Manual Actions */}
    <div className="p-4 border-t border-white/10 space-y-3 bg-black/20">
      <h3 className="text-[10px] font-bold text-white/30 uppercase tracking-widest flex items-center gap-2">
        Ações Manuais
      </h3>
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("ocr")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-2 px-3 py-2.5 bg-indigo-500/10 hover:bg-indigo-500/20 disabled:opacity-50 text-[11px] font-medium text-indigo-300 rounded border border-indigo-500/20 transition-all active:scale-95"
          title="Detecta o texto original dentro da área selecionada"
        >
          <Search className="w-3.5 h-3.5" />
          Ler Texto
        </button>
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("inpaint")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-2 px-3 py-2.5 bg-rose-500/10 hover:bg-rose-500/20 disabled:opacity-50 text-[11px] font-medium text-rose-300 rounded border border-rose-500/20 transition-all active:scale-95"
          title="Remove o texto original e limpa o balão (Inpainting)"
        >
          <div className="w-3.5 h-3.5 border-2 border-rose-300/50 rounded-full" />
          Limpar
        </button>
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("translate")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-2 px-3 py-2.5 bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-50 text-[11px] font-medium text-emerald-300 rounded border border-emerald-500/20 transition-all active:scale-95"
          title="Traduz o texto original para o idioma de destino"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRetypesetting ? "animate-spin" : ""}`} />
          Traduzir
        </button>
      </div>

      {/* Only show disconnect if it looks like a connected block */}
      {selectedLayer?.layout_group_size && selectedLayer.layout_group_size > 1 && (
        <button
          onClick={() => useEditorStore.getState().disconnectBlock()}
          className="w-full flex items-center justify-center gap-2 px-3 py-2.5 bg-rose-500/10 hover:bg-rose-500/20 text-[11px] font-medium text-rose-300 rounded border border-rose-500/20 transition-all active:scale-95 mt-2"
        >
          <Link2Off className="w-3.5 h-3.5" />
          Desconectar de Grupo
        </button>
      )}
    </div>
  </div>
);
}
