import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  Eye,
  EyeOff,
  ChevronLeft,
  FileText,
} from "lucide-react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../lib/stores/appStore";
import { exportProject, openExportDialog, openLogSaveDialog, exportTextFile } from "../lib/tauri";
import { createPsdFromLayers } from "../lib/psd";
import { writeFile } from "@tauri-apps/plugin-fs";

export function Preview() {
  const navigate = useNavigate();
  const { project } = useAppStore();
  const [currentPage, setCurrentPage] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [exportFormat, setExportFormat] = useState<"zip_full" | "jpg_only" | "cbz" | "psd">("zip_full");
  const [exporting, setExporting] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const prevBlobRef = useRef<string | null>(null);

  const totalPages = project?.paginas.length || 0;
  const page = project?.paginas[currentPage];

  useEffect(() => {
    if (!page) {
      if (prevBlobRef.current) {
        URL.revokeObjectURL(prevBlobRef.current);
        prevBlobRef.current = null;
      }
      setImageSrc(null);
      return;
    }
    const filePath = showOriginal ? page.arquivo_original : page.arquivo_traduzido;
    let cancelled = false;

    const loadImage = (path: string) =>
      readFile(path).then((bytes) => {
        if (cancelled) return;
        const blob = new Blob([bytes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current);
        prevBlobRef.current = url;
        setImageSrc(url);
      });

    loadImage(filePath).catch(() => {
      if (cancelled) return;
      // Fallback: se o translated não existe, tentar images/ (inpainted) depois originals/
      if (!showOriginal && page.arquivo_traduzido) {
        const inpaintedPath = page.arquivo_traduzido.replace("/translated/", "/images/");
        loadImage(inpaintedPath).catch(() => {
          if (cancelled) return;
          // Último fallback: mostrar o original
          loadImage(page.arquivo_original).catch(() => setImageSrc(null));
        });
      } else {
        setImageSrc(null);
      }
    });
    return () => { cancelled = true; };
  }, [page, showOriginal]);

  useEffect(() => {
    return () => {
      if (prevBlobRef.current) {
        URL.revokeObjectURL(prevBlobRef.current);
      }
    };
  }, []);

  async function handleExport() {
    if (!project) return;
    setExporting(true);
    try {
      const outputPath = await openExportDialog(exportFormat);
      if (!outputPath) return;

      if (exportFormat === "psd") {
        // Lógica de exportação PSD por página (mais robusta)
        for (let i = 0; i < project.paginas.length; i++) {
          const pg = project.paginas[i];
          const inpaintPath = pg.arquivo_traduzido.replace("/translated/", "/images/");
          
          // Criar canvas de texto fictício para agora (v0.50 Alpha)
          // Em v0.51 faremos a renderização real do canvas de texto aqui
          const dummyCanvas = document.createElement('canvas');
          dummyCanvas.width = 100; dummyCanvas.height = 100; 

          const psdData = await createPsdFromLayers(
            pg.arquivo_original,
            inpaintPath,
            dummyCanvas, // Futuro: Canvas com os textos renderizados
            100, 100     // Futuro: Dimensões reais da página
          );

          const fileName = pg.arquivo_original.split(/[/\\]/).pop()?.replace(/\.\w+$/, ".psd") || `pg-${pg.numero}.psd`;
          await writeFile(`${outputPath}/${fileName}`, psdData);
        }
      } else {
        await exportProject({
          project_path: project.output_path ?? project.source_path,
          format: exportFormat,
          output_path: outputPath,
        });
      }

      alert("Exportação concluída!");
    } catch (err) {
      console.error("Erro na exportação:", err);
      alert("Erro ao exportar.");
    } finally {
      setExporting(false);
    }
  }

  async function handleExportLog() {
    if (!project || !project.output_path) return;
    try {
      const logPath = `${project.output_path}/pipeline.log`.replace(/\\/g, "/");
      const contents = await readFile(logPath);
      const text = new TextDecoder().decode(contents);
      
      const savePath = await openLogSaveDialog(`log-${project.obra}-${project.capitulo}.log`);
      if (!savePath) return;

      await exportTextFile(savePath, text);
      alert("Log exportado com sucesso!");
    } catch (err) {
      console.error("Erro ao exportar log:", err);
      alert("O arquivo de log ainda não foi gerado ou não pôde ser lido.");
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-white/5 bg-bg-secondary">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/")}
            title="Voltar para o início"
            className="p-1.5 text-text-secondary hover:text-text-primary transition-smooth"
          >
            <ChevronLeft size={18} />
          </button>
          <div>
            <p className="text-sm font-medium">{project?.obra}</p>
            <p className="text-xs text-text-secondary">
              Capítulo {project?.capitulo} — {totalPages} páginas
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Toggle original/translated */}
          <button
            onClick={() => setShowOriginal(!showOriginal)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-smooth
              ${showOriginal
                ? "bg-status-warning/10 text-status-warning border border-status-warning/20"
                : "bg-bg-tertiary text-text-secondary border border-white/5"
              }`}
          >
            {showOriginal ? <EyeOff size={14} /> : <Eye size={14} />}
            {showOriginal ? "Original" : "Traduzido"}
          </button>

          {/* Editor button */}
          <button
            onClick={() => navigate("/editor")}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-purple text-white
              rounded-lg text-xs font-medium hover:bg-accent-purple-dark transition-smooth"
          >
            Abrir Editor
          </button>

          {/* Export button */}
          <button
            onClick={() => setShowExportPanel(!showExportPanel)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-purple/10 text-accent-purple
              rounded-lg text-xs hover:bg-accent-purple/20 transition-smooth"
          >
            <Download size={14} />
            Exportar
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Image viewer */}
        <div className="flex-1 flex items-center justify-center p-4 bg-bg-primary">
          {page && imageSrc ? (
            <div className="max-h-full max-w-full relative">
              <img
                src={imageSrc}
                alt={`Página ${page.numero}`}
                className="max-h-[calc(100vh-140px)] w-auto object-contain rounded-lg shadow-2xl"
              />
            </div>
          ) : page ? (
            <p className="text-text-secondary text-sm">Carregando imagem...</p>
          ) : (
            <p className="text-text-secondary">Nenhuma página para exibir</p>
          )}
        </div>

        {/* Export panel (slide-in) */}
        {showExportPanel && (
          <div className="w-72 bg-bg-secondary border-l border-white/5 p-5 space-y-4">
            <h3 className="text-sm font-medium">Exportar projeto</h3>

            <div className="space-y-2">
              {(
                [
                  { value: "zip_full", label: "ZIP completo", desc: "Originais + traduzidas + project.json" },
                  { value: "jpg_only", label: "Somente traduzidas", desc: "Apenas as imagens traduzidas" },
                  { value: "cbz", label: "CBZ", desc: "Formato de leitor de mangá" },
                  { value: "psd", label: "Photoshop (PSD)", desc: "Camadas separadas: Original, Inpaint, Texto" },
                ] as const
              ).map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setExportFormat(opt.value)}
                  className={`w-full text-left p-3 rounded-lg border transition-smooth
                    ${exportFormat === opt.value
                      ? "border-accent-purple/40 bg-accent-purple/5"
                      : "border-white/5 hover:border-white/10"
                    }`}
                >
                  <p className="text-sm font-medium">{opt.label}</p>
                  <p className="text-xs text-text-secondary mt-0.5">{opt.desc}</p>
                </button>
              ))}
            </div>

            <div className="pt-2 space-y-2">
              <button
                onClick={handleExport}
                disabled={exporting}
                className="w-full py-2.5 bg-accent-purple hover:bg-accent-purple-dark text-white
                  text-sm font-medium rounded-lg transition-smooth disabled:opacity-50"
              >
                {exporting ? "Exportando..." : "Salvar arquivo"}
              </button>

              <button
                onClick={handleExportLog}
                className="w-full py-2 flex items-center justify-center gap-2 bg-bg-tertiary 
                  hover:bg-white/5 text-text-secondary hover:text-text-primary text-xs font-medium 
                  rounded-lg border border-white/5 transition-smooth"
              >
                <FileText size={14} />
                Exportar Log do Pipeline
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Page navigation */}
      <div className="flex items-center justify-center gap-4 px-6 py-3 border-t border-white/5 bg-bg-secondary">
        <button
          onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
          disabled={currentPage === 0}
          title="Página anterior"
          className="p-2 rounded-lg bg-bg-tertiary text-text-secondary hover:text-text-primary
            transition-smooth disabled:opacity-30"
        >
          <ArrowLeft size={16} />
        </button>

        <span className="text-sm font-mono text-text-secondary min-w-[80px] text-center">
          {currentPage + 1} / {totalPages}
        </span>

        <button
          onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
          disabled={currentPage >= totalPages - 1}
          title="Próxima página"
          className="p-2 rounded-lg bg-bg-tertiary text-text-secondary hover:text-text-primary
            transition-smooth disabled:opacity-30"
        >
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
