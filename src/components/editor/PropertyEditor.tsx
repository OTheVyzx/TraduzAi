import { useAppStore } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";
import {
  AlignLeft,
  AlignCenter,
  AlignRight,
  Bold,
  Italic,
  ChevronDown,
  ChevronRight,
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
}: {
  value: string | number;
  onChange: (val: string) => void;
  type?: "text" | "number";
  readOnly?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      readOnly={readOnly}
      className={`w-full px-2 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm text-text-primary
        focus:border-accent-purple/50 focus:outline-none transition-smooth
        ${readOnly ? "opacity-60 cursor-default" : ""}`}
    />
  );
}

export function PropertyEditor() {
  const project = useAppStore((s) => s.project);
  const { selectedLayerId, currentPageIndex, pendingEdits } = useEditorStore();
  const updateEdit = useEditorStore((s) => s.updatePendingEdit);
  const updateEstilo = useEditorStore((s) => s.updatePendingEstilo);

  const page = project?.paginas[currentPageIndex];
  const entry = page?.textos.find((t) => t.id === selectedLayerId);

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
  const traduzido = edit?.traduzido ?? entry.traduzido;
  const original = entry.original;
  const estilo = edit?.estilo ? { ...entry.estilo, ...edit.estilo } : entry.estilo;
  const [x1, y1, x2, y2] = edit?.bbox ?? entry.bbox;

  const confidencePercent = Math.round((entry.confianca_ocr ?? 0) * 100);
  const confidenceColor =
    confidencePercent >= 80
      ? "text-status-success"
      : confidencePercent >= 50
        ? "text-status-warning"
        : "text-status-error";

  return (
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
            onChange={(e) => updateEdit(selectedLayerId, { traduzido: e.target.value })}
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
  );
}
