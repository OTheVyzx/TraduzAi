import { useEffect, useRef, useState } from "react";
import { loadImageSource } from "../../lib/imageSource";
import { useAppStore } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";

function resolveProjectImagePath(
  path: string,
  project: { output_path?: string | null; source_path?: string | null; _work_dir?: string | null },
) {
  const normalized = path.replace(/\\/g, "/");
  if (
    /^[A-Za-z]:\//.test(normalized) ||
    normalized.startsWith("/") ||
    /^(data|blob|asset|file):/i.test(normalized) ||
    /^https?:\/\//i.test(normalized)
  ) {
    return normalized;
  }
  const base = (project.output_path || project.source_path || project._work_dir || "")
    .replace(/\\/g, "/")
    .replace(/\/project\.json$/i, "");
  return base ? `${base}/${normalized}` : normalized;
}

function Thumbnail({
  path,
  numero,
  pageIndex,
  blocks,
  isActive,
  onClick,
}: {
  path: string;
  numero: number;
  pageIndex: number;
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
    let revokeSource: (() => void) | null = null;

    loadImageSource(path, "image/jpeg")
      .then((loaded) => {
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        revokeSource = loaded.revoke ?? null;
        setSrc(loaded.src);
      })
      .catch((error) => console.error("Falha ao ler thumbnail", path, error));

    return () => {
      cancelled = true;
      revokeSource?.();
    };
  }, [path, isVisible]);

  return (
    <button
      ref={containerRef}
      data-testid={`editor-page-thumbnail-${pageIndex + 1}`}
      onClick={onClick}
      className={`group relative w-full flex-shrink-0 overflow-hidden rounded-lg text-left transition-all duration-150 ${
        isActive
          ? "ring-2 ring-brand ring-offset-1 ring-offset-bg-primary shadow-[0_0_12px_rgba(108,92,231,0.2)]"
          : "ring-1 ring-transparent hover:ring-white/15"
      }`}
    >
      <div className="aspect-[2/3] w-full overflow-hidden bg-bg-tertiary rounded-lg">
        {src ? (
          <img
            src={src}
            alt={`Pagina ${numero}`}
            className={`h-full w-full object-cover transition-all duration-150 ${
              isActive ? "" : "opacity-70 group-hover:opacity-100"
            }`}
          />
        ) : (
          <div className="h-full w-full animate-pulse bg-bg-tertiary" />
        )}
      </div>

      {/* Page number overlay */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-2 pb-1.5 pt-4">
        <div className="flex items-center justify-between">
          <span className={`text-[10px] font-semibold ${isActive ? "text-brand-300" : "text-white/80"}`}>
            {numero}
          </span>
          <span className="text-[9px] text-white/50">{blocks}</span>
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
    <div className="flex w-[100px] flex-shrink-0 flex-col border-r border-border bg-bg-primary">
      <div className="border-b border-border px-3 py-2.5">
        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-text-muted">
          Paginas
        </p>
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-2">
        <div className="flex flex-col gap-2">
          {project.paginas.map((page, index) => {
            const rawThumbPath =
              viewMode === "translated"
                ? page.image_layers?.rendered?.path || page.arquivo_traduzido || page.arquivo_original
                : viewMode === "inpainted"
                  ? page.image_layers?.inpaint?.path || page.arquivo_original
                  : page.image_layers?.base?.path || page.arquivo_original;
            const thumbPath = resolveProjectImagePath(rawThumbPath, project);
            return (
              <Thumbnail
                key={`${page.numero}-${thumbPath}`}
                numero={page.numero}
                pageIndex={index}
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
