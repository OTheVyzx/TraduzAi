import { useEffect, useRef, useState } from "react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";

function Thumbnail({
  path,
  numero,
  blocks,
  isActive,
  onClick,
}: {
  path: string;
  numero: number;
  blocks: number;
  isActive: boolean;
  onClick: () => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;
    containerRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [isActive]);

  useEffect(() => {
    if (!isVisible) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    readFile(path)
      .then((bytes) => {
        if (cancelled) return;
        const blob = new Blob([bytes], { type: "image/jpeg" });
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch((error) => console.error("Falha ao ler thumbnail", path, error));

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path, isVisible]);

  return (
    <button
      ref={containerRef}
      onClick={onClick}
      className={`group relative w-full flex-shrink-0 overflow-hidden rounded-xl border-2 text-left transition-smooth ${
        isActive
          ? "border-accent-purple shadow-[0_0_0_1px_rgba(124,92,255,0.25)]"
          : "border-transparent hover:border-white/20"
      }`}
    >
      <div className="aspect-[2/3] w-full overflow-hidden bg-bg-tertiary">
        {src ? (
          <img src={src} alt={`Pagina ${numero}`} className="h-full w-full object-cover" />
        ) : (
          <div className="h-full w-full animate-pulse bg-bg-tertiary" />
        )}
      </div>

      <div className="border-t border-white/5 bg-bg-secondary/95 px-2 py-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className={`font-medium ${isActive ? "text-accent-purple" : "text-text-primary"}`}>
            Pag. {numero}
          </span>
          <span className="text-text-muted">{blocks} bloco(s)</span>
        </div>
      </div>
    </button>
  );
}

export function PageThumbnails() {
  const project = useAppStore((s) => s.project);
  const { currentPageIndex, setCurrentPage, viewMode } = useEditorStore();

  if (!project) return null;

  return (
    <div className="flex w-[124px] flex-shrink-0 flex-col border-r border-white/5 bg-bg-secondary">
      <div className="border-b border-white/5 px-3 py-3">
        <p className="text-[11px] uppercase tracking-[0.18em] text-text-muted">Paginas</p>
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-2.5">
        <div className="flex flex-col gap-2">
          {project.paginas.map((page, index) => {
            const thumbPath =
              viewMode === "translated"
                ? page.image_layers?.rendered?.path || page.arquivo_traduzido || page.arquivo_original
                : viewMode === "inpainted"
                  ? page.image_layers?.inpaint?.path || page.arquivo_original
                  : page.image_layers?.base?.path || page.arquivo_original;
            return (
              <Thumbnail
                key={`${page.numero}-${thumbPath}`}
                numero={page.numero}
                path={thumbPath}
                blocks={(page.text_layers ?? page.textos).length}
                isActive={index === currentPageIndex}
                onClick={() => void setCurrentPage(index)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
