import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  Check,
  Download,
  Edit3,
  Eye,
  EyeOff,
  FileText,
  LocateFixed,
  Minus,
  Plus,
  RotateCw,
  ShieldPlus,
} from "lucide-react";
import { readFile } from "@tauri-apps/plugin-fs";
import { loadImageSource, preloadImageSource, type LoadedImageSource } from "../lib/imageSource";
import { getPageKey } from "../lib/editorHistory";
import { useAppStore, type PageData, type Project } from "../lib/stores/appStore";
import {
  buildQaReviewSummary,
  collectIgnoredQaActions,
  collectQaIssues,
  ignoreQaIssue,
  qaIssueGroup,
  type QaIssue,
} from "../lib/qaPanel";
import { useEditorStore, type RenderPreviewCacheByPageKey } from "../lib/stores/editorStore";
import {
  exportPagePsd,
  exportProject,
  exportTextFile,
  openExportDialog,
  openLogSaveDialog,
} from "../lib/tauri";
import {
  getDraggedPreviewPan,
  getNextPreviewZoom,
  getPreviewWheelState,
  PREVIEW_ZOOM_DEFAULT,
  type PreviewPanOffset,
  type PreviewPanSession,
} from "./previewZoom";
import { getPreviewImageCandidates, getPreviewToggleLabel } from "./previewImage";
import { EXPORT_MODE_OPTIONS, exportBlockReason, exportModeForBackend, type ExportMode } from "../lib/exportModes";
import { renderPageWithKonvaToDataUrl, shouldUseKonvaPreviewRenderer } from "../lib/konvaExportRenderer";

function waitForImageLoad(src: string) {
  return new Promise<void>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("imagem indisponivel"));
    image.src = src;
  });
}

type PreviewScrollAxis = "vertical" | "horizontal";
type PreviewReaderImage = {
  pageIndex: number;
  pageNumber: number;
  src: string | null;
  status: "loading" | "ready" | "error";
};

function readerPageLoadOrder<T>(pages: T[], currentPage: number) {
  return pages
    .map((readerPage, pageIndex) => [pageIndex, readerPage] as const)
    .sort(([leftIndex], [rightIndex]) => {
      const leftDistance = Math.abs(leftIndex - currentPage);
      const rightDistance = Math.abs(rightIndex - currentPage);
      return leftDistance - rightDistance || leftIndex - rightIndex;
    });
}

function getPreviewReaderImageCandidates({
  page,
  pageIndex,
  project,
  projectImageBasePath,
  renderPreviewCacheByPageKey,
  showOriginal,
}: {
  page: PageData;
  pageIndex: number;
  project: Project;
  projectImageBasePath: string | null;
  renderPreviewCacheByPageKey: RenderPreviewCacheByPageKey;
  showOriginal: boolean;
}) {
  const readerPageKey = getPageKey(project, pageIndex);
  const readerPreviewState = renderPreviewCacheByPageKey[readerPageKey];
  const readerFaithfulPreviewPath =
    !showOriginal && readerPreviewState?.status === "fresh" ? readerPreviewState.previewPath : null;
  return getPreviewImageCandidates(
    page,
    showOriginal,
    projectImageBasePath,
    readerFaithfulPreviewPath,
  );
}

async function loadPreviewReaderImage({
  page,
  pageIndex,
  project,
  projectImageBasePath,
  renderPreviewCacheByPageKey,
  showOriginal,
  useKonvaPreviewRenderer,
}: {
  page: PageData;
  pageIndex: number;
  project: Project;
  projectImageBasePath: string | null;
  renderPreviewCacheByPageKey: RenderPreviewCacheByPageKey;
  showOriginal: boolean;
  useKonvaPreviewRenderer: boolean;
}): Promise<LoadedImageSource> {
  if (useKonvaPreviewRenderer) {
    try {
      const src = await renderPageWithKonvaToDataUrl({ page, projectImageBasePath });
      await waitForImageLoad(src);
      return { src };
    } catch {
      // O render fiel pode falhar em paginas muito altas ou em assets locais;
      // nesses casos o preview precisa cair para o JPG final ja gerado.
    }
  }

  const candidatePaths = getPreviewReaderImageCandidates({
    page,
    pageIndex,
    project,
    projectImageBasePath,
    renderPreviewCacheByPageKey,
    showOriginal,
  });

  let lastError: unknown = null;
  for (const candidatePath of candidatePaths) {
    let loaded: LoadedImageSource | null = null;
    try {
      loaded = await loadImageSource(candidatePath, "image/jpeg");
      await waitForImageLoad(loaded.src);
      return loaded;
    } catch (error) {
      lastError = error;
      loaded?.revoke?.();
    }
  }

  throw lastError instanceof Error ? lastError : new Error("imagem indisponivel");
}

async function preloadPreviewReaderImage({
  page,
  pageIndex,
  project,
  projectImageBasePath,
  renderPreviewCacheByPageKey,
  showOriginal,
}: {
  page: PageData;
  pageIndex: number;
  project: Project;
  projectImageBasePath: string | null;
  renderPreviewCacheByPageKey: RenderPreviewCacheByPageKey;
  showOriginal: boolean;
}) {
  const candidatePaths = getPreviewReaderImageCandidates({
    page,
    pageIndex,
    project,
    projectImageBasePath,
    renderPreviewCacheByPageKey,
    showOriginal,
  });

  for (const candidatePath of candidatePaths) {
    try {
      await preloadImageSource(candidatePath, "image/jpeg");
      return;
    } catch {
      // Tenta o proximo candidato; o fluxo normal ainda faz fallback ao exibir.
    }
  }
}

export function Preview() {
  const navigate = useNavigate();
  const { project, updateProject, batchCompletion } = useAppStore();
  const renderPreviewCacheByPageKey = useEditorStore((s) => s.renderPreviewCacheByPageKey);
  const renderPreviewPageForPage = useEditorStore((s) => s.renderPreviewPageForPage);
  const getStaleRenderPreviewPages = useEditorStore((s) => s.getStaleRenderPreviewPages);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const pendingStructuralEdits = useEditorStore((s) => s.pendingStructuralEdits);
  const [currentPage, setCurrentPage] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [exportFormat, setExportFormat] = useState<"zip_full" | "jpg_only" | "cbz" | "psd">("zip_full");
  const [exportMode, setExportMode] = useState<ExportMode>("clean");
  const [exporting, setExporting] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [zoom, setZoom] = useState(PREVIEW_ZOOM_DEFAULT);
  const [panOffset, setPanOffset] = useState<PreviewPanOffset>({ x: 0, y: 0 });
  const [panSession, setPanSession] = useState<PreviewPanSession | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [scrollAxis, setScrollAxis] = useState<PreviewScrollAxis>("vertical");
  const [readerImages, setReaderImages] = useState<PreviewReaderImage[]>([]);
  const [ignoreIssueId, setIgnoreIssueId] = useState<string | null>(null);
  const [ignoreReason, setIgnoreReason] = useState("");
  const [ignoreError, setIgnoreError] = useState<string | null>(null);
  const [lastIgnoredReason, setLastIgnoredReason] = useState<string | null>(null);
  const [exportBlockMessage, setExportBlockMessage] = useState<string | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const prevImageRevokeRef = useRef<(() => void) | null>(null);
  const readerRevokesRef = useRef<(() => void)[]>([]);
  const readerPriorityLoadsRef = useRef<Set<number>>(new Set());
  const readerPageRefs = useRef<Array<HTMLDivElement | null>>([]);
  const scrollToPageRef = useRef(false);
  const scrollSyncFrameRef = useRef<number | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const totalPages = project?.paginas.length || 0;
  const page = project?.paginas[currentPage] ?? null;
  const pageKey = project ? getPageKey(project, currentPage) : "";
  const renderPreviewState = pageKey ? renderPreviewCacheByPageKey[pageKey] : null;
  const useKonvaPreviewRenderer = !showOriginal && shouldUseKonvaPreviewRenderer();
  const faithfulPreviewPath =
    !useKonvaPreviewRenderer && renderPreviewState?.status === "fresh" ? renderPreviewState.previewPath : null;
  const projectImageBasePath = project
    ? project.output_path ||
      project.source_path ||
      (project as typeof project & { _work_dir?: string | null })._work_dir ||
      null
    : null;
  const staleRenderPages = useKonvaPreviewRenderer ? [] : getStaleRenderPreviewPages();
  const hasUncommittedEditorEdits =
    Object.keys(pendingEdits).length > 0 ||
    pendingStructuralEdits.created.length > 0 ||
    Object.keys(pendingStructuralEdits.deleted).length > 0 ||
    Boolean(pendingStructuralEdits.order);
  const qaIssues = collectQaIssues(project);
  const ignoredQaActions = collectIgnoredQaActions(project);
  const qaReviewSummary = buildQaReviewSummary(project);
  const activeIgnoreIssue = qaIssues.find((issue) => issue.id === ignoreIssueId) ?? null;
  const qaGroups = Object.entries(qaReviewSummary.groups);

  const preloadPreviewPage = (pageIndex: number) => {
    if (!project || pageIndex < 0 || pageIndex >= project.paginas.length) return;
    const targetPage = project.paginas[pageIndex];
    void preloadPreviewReaderImage({
      page: targetPage,
      pageIndex,
      project,
      projectImageBasePath,
      renderPreviewCacheByPageKey,
      showOriginal,
    }).catch(() => {});
  };

  useEffect(() => {
    if (!project?.paginas.length) return;
    const indexes = [
      currentPage,
      currentPage - 2,
      currentPage - 1,
      currentPage + 1,
      currentPage + 2,
    ];
    for (const index of indexes) {
      preloadPreviewPage(index);
    }
  }, [currentPage, project, project?.paginas, projectImageBasePath, renderPreviewCacheByPageKey, showOriginal]);

  useEffect(() => {
    if (!project || !page || showOriginal || !pageKey || useKonvaPreviewRenderer) return;
    const cached = renderPreviewCacheByPageKey[pageKey];
    if (cached?.status === "rendering" || cached?.status === "error") return;
    if (cached?.status === "fresh" && cached.previewPath) return;

    void renderPreviewPageForPage(pageKey, currentPage, page).catch(() => {
      // O JPG final antigo continua como fallback visual se a renderizacao fiel falhar.
    });
  }, [currentPage, page, pageKey, project, renderPreviewCacheByPageKey, renderPreviewPageForPage, showOriginal, useKonvaPreviewRenderer]);

  useEffect(() => {
    if (!page) {
      prevImageRevokeRef.current?.();
      prevImageRevokeRef.current = null;
      setImageSrc(null);
      return;
    }

    let cancelled = false;
    const candidatePaths = getPreviewImageCandidates(page, showOriginal, projectImageBasePath, faithfulPreviewPath);
    setImageSrc(null);

    const loadFirstAvailable = async () => {
      if (useKonvaPreviewRenderer) {
        try {
          const src = await renderPageWithKonvaToDataUrl({ page, projectImageBasePath });
          if (!cancelled) {
            prevImageRevokeRef.current?.();
            prevImageRevokeRef.current = null;
            setImageSrc(src);
            return;
          }
        } catch {
          if (cancelled) return;
        }
      }
      for (const candidatePath of candidatePaths) {
        let loaded: Awaited<ReturnType<typeof loadImageSource>> | null = null;
        try {
          loaded = await loadImageSource(candidatePath, "image/jpeg");
          if (cancelled) {
            loaded.revoke?.();
            return;
          }
          await waitForImageLoad(loaded.src);
          if (cancelled) {
            loaded.revoke?.();
            return;
          }
          prevImageRevokeRef.current?.();
          prevImageRevokeRef.current = loaded.revoke ?? null;
          setImageSrc(loaded.src);
          return;
        } catch {
          loaded?.revoke?.();
          if (cancelled) return;
        }
      }

      if (!cancelled) {
        prevImageRevokeRef.current?.();
        prevImageRevokeRef.current = null;
        setImageSrc(null);
      }
    };

    void loadFirstAvailable();

    return () => {
      cancelled = true;
    };
  }, [page, showOriginal, projectImageBasePath, faithfulPreviewPath, useKonvaPreviewRenderer]);

  useEffect(() => {
    return () => {
      prevImageRevokeRef.current?.();
      if (scrollSyncFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollSyncFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    readerRevokesRef.current.forEach((revoke) => revoke());
    readerRevokesRef.current = [];

    if (scrollAxis !== "vertical" || !project?.paginas.length) {
      setReaderImages([]);
      return;
    }

    let cancelled = false;
    setReaderImages(
      project.paginas.map((readerPage, pageIndex) => ({
        pageIndex,
        pageNumber: readerPage.numero,
        src: null,
        status: "loading",
      })),
    );

    const loadReaderImages = async () => {
      for (const [pageIndex, readerPage] of readerPageLoadOrder(project.paginas, currentPage)) {
        let loaded: Awaited<ReturnType<typeof loadImageSource>> | null = null;
        try {
          loaded = await loadPreviewReaderImage({
            page: readerPage,
            pageIndex,
            project,
            projectImageBasePath,
            renderPreviewCacheByPageKey,
            showOriginal,
            useKonvaPreviewRenderer,
          });

          if (cancelled) {
            loaded?.revoke?.();
            return;
          }

          if (loaded?.src) {
            const readerSrc = loaded.src;
            if (loaded.revoke) readerRevokesRef.current.push(loaded.revoke);
            setReaderImages((current) =>
              current.map((item) =>
                item.pageIndex === pageIndex ? { ...item, src: readerSrc, status: "ready" } : item,
              ),
            );
          } else {
            setReaderImages((current) =>
              current.map((item) => (item.pageIndex === pageIndex ? { ...item, status: "error" } : item)),
            );
          }
        } catch {
          loaded?.revoke?.();
          if (!cancelled) {
            setReaderImages((current) =>
              current.map((item) => (item.pageIndex === pageIndex ? { ...item, status: "error" } : item)),
            );
          }
        }
      }
    };

    void loadReaderImages();

    return () => {
      cancelled = true;
      readerPriorityLoadsRef.current.clear();
      readerRevokesRef.current.forEach((revoke) => revoke());
      readerRevokesRef.current = [];
    };
  }, [project, project?.paginas, projectImageBasePath, renderPreviewCacheByPageKey, scrollAxis, showOriginal, useKonvaPreviewRenderer]);

  useEffect(() => {
    if (scrollAxis !== "vertical" || !project?.paginas.length) return;
    const readerPage = project.paginas[currentPage];
    const readerItem = readerImages.find((item) => item.pageIndex === currentPage);
    if (!readerPage || !readerItem || readerItem.status !== "loading" || readerPriorityLoadsRef.current.has(currentPage)) {
      return;
    }

    let cancelled = false;
    readerPriorityLoadsRef.current.add(currentPage);

    const loadCurrentReaderImage = async () => {
      let loaded: LoadedImageSource | null = null;
      try {
        loaded = await loadPreviewReaderImage({
          page: readerPage,
          pageIndex: currentPage,
          project,
          projectImageBasePath,
          renderPreviewCacheByPageKey,
          showOriginal,
          useKonvaPreviewRenderer,
        });
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        if (loaded.revoke) readerRevokesRef.current.push(loaded.revoke);
        setReaderImages((current) =>
          current.map((item) =>
            item.pageIndex === currentPage ? { ...item, src: loaded?.src ?? item.src, status: "ready" } : item,
          ),
        );
      } catch {
        loaded?.revoke?.();
        if (!cancelled) {
          setReaderImages((current) =>
            current.map((item) => (item.pageIndex === currentPage ? { ...item, status: "error" } : item)),
          );
        }
      } finally {
        readerPriorityLoadsRef.current.delete(currentPage);
      }
    };

    void loadCurrentReaderImage();

    return () => {
      cancelled = true;
      readerPriorityLoadsRef.current.delete(currentPage);
    };
  }, [
    currentPage,
    project,
    project?.paginas,
    projectImageBasePath,
    readerImages,
    renderPreviewCacheByPageKey,
    scrollAxis,
    showOriginal,
    useKonvaPreviewRenderer,
  ]);

  useEffect(() => {
    if (
      scrollAxis !== "vertical" ||
      showOriginal ||
      !faithfulPreviewPath ||
      useKonvaPreviewRenderer
    ) return;

    let cancelled = false;
    const loadFaithfulReaderPage = async () => {
      let loaded: Awaited<ReturnType<typeof loadImageSource>> | null = null;
      try {
        loaded = await loadImageSource(faithfulPreviewPath, "image/jpeg");
        await waitForImageLoad(loaded.src);
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        if (loaded.revoke) readerRevokesRef.current.push(loaded.revoke);
        setReaderImages((current) =>
          current.map((item) =>
            item.pageIndex === currentPage ? { ...item, src: loaded?.src ?? item.src, status: "ready" } : item,
          ),
        );
      } catch {
        loaded?.revoke?.();
      }
    };

    void loadFaithfulReaderPage();

    return () => {
      cancelled = true;
    };
  }, [currentPage, faithfulPreviewPath, scrollAxis, showOriginal, useKonvaPreviewRenderer]);

  useEffect(() => {
    if (scrollAxis !== "vertical") return;
    if (!scrollToPageRef.current) return;
    scrollToPageRef.current = false;
    readerPageRefs.current[currentPage]?.scrollIntoView({ block: "start", inline: "nearest", behavior: "smooth" });
  }, [currentPage, readerImages.length, scrollAxis]);

  useEffect(() => {
    if (scrollAxis !== "vertical") return;
    setPanOffset({ x: 0, y: 0 });
    viewportRef.current?.scrollTo({ left: 0 });
  }, [scrollAxis]);

  useEffect(() => {
    if (scrollAxis !== "horizontal") return;
    setZoom(PREVIEW_ZOOM_DEFAULT);
    setPanOffset({ x: 0, y: 0 });
    setPanSession(null);
  }, [currentPage, scrollAxis]);

  useEffect(() => {
    setZoom(PREVIEW_ZOOM_DEFAULT);
    setPanOffset({ x: 0, y: 0 });
    setPanSession(null);
  }, [showOriginal]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const active = document.activeElement;
      const isTyping =
        !!active &&
        (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT");

      if (event.key === " ") {
        if (isTyping) return;
        event.preventDefault();
        setIsSpacePressed(true);
        return;
      }

      if (isTyping) return;

      if (event.key === "=" || event.key === "+") {
        event.preventDefault();
        setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "in"));
      }
      if (event.key === "-") {
        event.preventDefault();
        setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "out"));
      }
      if (event.key === "0") {
        event.preventDefault();
        resetPreviewView();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") setIsSpacePressed(false);
    };

    const handleBlur = () => {
      setIsSpacePressed(false);
      setPanSession(null);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  useEffect(() => {
    if (!panSession) return;

    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      setPanOffset(getDraggedPreviewPan(panSession, event));
    };

    const handleMouseUp = () => setPanSession(null);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panSession]);

  function resetPreviewView() {
    setZoom(PREVIEW_ZOOM_DEFAULT);
    setPanOffset({ x: 0, y: 0 });
    setPanSession(null);
  }

  function beginPan(clientX: number, clientY: number) {
    setPanSession({
      startX: clientX,
      startY: clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    });
  }

  function goToPreviewPage(index: number) {
    const nextPage = Math.max(0, Math.min(totalPages - 1, index));
    preloadPreviewPage(nextPage);
    scrollToPageRef.current = scrollAxis === "vertical";
    setCurrentPage(nextPage);
  }

  function syncCurrentPageFromScroll() {
    const viewport = viewportRef.current;
    if (!viewport || scrollAxis !== "vertical") return;

    const viewportRect = viewport.getBoundingClientRect();
    const viewportMiddleY = viewportRect.top + viewportRect.height / 2;
    let nearestPage = currentPage;
    let nearestDistance = Number.POSITIVE_INFINITY;

    for (const item of readerImages) {
      const node = readerPageRefs.current[item.pageIndex];
      if (!node) continue;
      const rect = node.getBoundingClientRect();
      if (rect.top <= viewportMiddleY && rect.bottom >= viewportMiddleY) {
        nearestPage = item.pageIndex;
        break;
      }
      const distance = Math.min(Math.abs(rect.top - viewportMiddleY), Math.abs(rect.bottom - viewportMiddleY));
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestPage = item.pageIndex;
      }
    }

    if (nearestPage !== currentPage) {
      setCurrentPage(nearestPage);
    }
  }

  function handleViewportScroll() {
    if (scrollAxis !== "vertical" || scrollSyncFrameRef.current !== null) return;
    scrollSyncFrameRef.current = window.requestAnimationFrame(() => {
      scrollSyncFrameRef.current = null;
      syncCurrentPageFromScroll();
    });
  }

  function handleViewportWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (!event.ctrlKey && !event.metaKey) {
      if (scrollAxis === "horizontal") {
        event.preventDefault();
        const node = viewportRef.current;
        if (node) node.scrollLeft += event.deltaX || event.deltaY;
      }
      return;
    }
    event.preventDefault();

    const nextState = getPreviewWheelState({
      zoom,
      pan: panOffset,
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      withZoomModifier: event.ctrlKey || event.metaKey,
    });

    setZoom(nextState.zoom);
    setPanOffset(nextState.pan);
  }

  function handleViewportMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      beginPan(event.clientX, event.clientY);
    }
  }

  async function handleExport(options: { mode?: ExportMode } = {}) {
    if (!project) return;
    setExportBlockMessage(null);
    const activeMode = options.mode ?? exportMode;
    const blockReason = exportBlockReason(activeMode, qaReviewSummary);
    if (blockReason) {
      setExportBlockMessage(blockReason);
      return;
    }
    if (staleRenderPages.length > 0) {
      if (hasUncommittedEditorEdits) {
        alert(
          `Preview final desatualizado nas paginas: ${staleRenderPages.join(", ")}. ` +
            "Salve as edicoes abertas no editor antes de exportar.",
        );
        return;
      }
      try {
        for (const pageNumber of staleRenderPages) {
          const pageIndex = pageNumber - 1;
          const stalePage = project.paginas[pageIndex];
          if (!stalePage) continue;
          const stalePageKey = getPageKey(project, pageIndex);
          const cached = renderPreviewCacheByPageKey[stalePageKey];
          if (cached?.status === "fresh" && cached.previewPath) continue;
          await renderPreviewPageForPage(stalePageKey, pageIndex, stalePage);
        }
      } catch (err) {
        alert(`Erro ao renderizar preview final: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    setExporting(true);

    try {
      const outputPath = await openExportDialog(exportFormat);
      if (!outputPath) return;

      if (exportFormat === "psd") {
        const projectPath = project.output_path || project.source_path;
        const baseOutputPath = outputPath.replace(/[/\\][^/\\]+$/, "");

        for (let i = 0; i < project.paginas.length; i++) {
          const currentProjectPage = project.paginas[i];
          const fileName =
            currentProjectPage.arquivo_original
              .split(/[/\\]/)
              .pop()
              ?.replace(/\.\w+$/, ".psd") || `pg-${currentProjectPage.numero}.psd`;
          const finalPath = `${baseOutputPath}/${fileName}`.replace(/\\/g, "/");

          await exportPagePsd({
            project_path: projectPath,
            page_index: i,
            output_path: finalPath,
          });
        }
      } else {
        await exportProject({
          project_path: project.output_path || project.source_path,
          format: exportFormat,
          output_path: outputPath,
          export_mode: exportModeForBackend(activeMode),
        });
      }

      alert("Exportacao concluida!");
    } catch (err) {
      console.error("Erro na exportacao:", err);
      alert(`Erro ao exportar: ${err instanceof Error ? err.message : String(err)}`);
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
      alert("O arquivo de log ainda nao foi gerado ou nao pode ser lido.");
    }
  }

  function goToQaIssue(issue: QaIssue) {
    goToPreviewPage(issue.pageIndex);
  }

  function startIgnoreIssue(issue: QaIssue) {
    setIgnoreIssueId(issue.id);
    setIgnoreReason("");
    setIgnoreError(null);
  }

  function confirmIgnoreIssue() {
    if (!project || !ignoreIssueId) return;
    try {
      const updatedProject = ignoreQaIssue(project, ignoreIssueId, ignoreReason);
      updateProject({ paginas: updatedProject.paginas });
      setLastIgnoredReason(ignoreReason.trim());
      setIgnoreIssueId(null);
      setIgnoreReason("");
      setIgnoreError(null);
    } catch (err) {
      setIgnoreError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExportQaReport() {
    if (!project) return;
    const report = [
      `# QA - ${project.obra}`,
      "",
      `Capitulo: ${project.capitulo}`,
      `Paginas: ${qaReviewSummary.totalPages}`,
      `Aprovadas: ${qaReviewSummary.approvedPages}`,
      `Com aviso: ${qaReviewSummary.warningPages}`,
      `Bloqueadas: ${qaReviewSummary.blockedPages}`,
      `Flags ativas: ${qaIssues.length}`,
      `Ignoradas: ${ignoredQaActions.length}`,
      "",
      "## Grupos",
      ...qaGroups.map(([group, count]) => `- ${group}: ${count}`),
      "",
      "## Flags",
      ...qaIssues.map(
        (issue) =>
          `- Pagina ${issue.pageNumber} / ${issue.regionId}: ${issue.label} (${issue.severity}) - ${issue.sourceText}`,
      ),
      "",
      "## Acoes do usuario",
      ...ignoredQaActions.map(
        (action) => `- ${action.flag_id}: ignorado em ${action.ignored_at ?? "-"} - ${action.ignored_reason ?? "-"}`,
      ),
      "",
    ].join("\n");

    const savePath = await openLogSaveDialog(`qa-${project.obra}-${project.capitulo}.md`);
    if (!savePath) return;
    await exportTextFile(savePath, report);
  }

  const viewportCursor = panSession ? "cursor-grabbing" : isSpacePressed ? "cursor-grab" : "cursor-default";
  const viewportOverflowClass =
    scrollAxis === "vertical" ? "overflow-y-auto overflow-x-hidden" : "overflow-x-auto overflow-y-hidden";
  const previewFrameClass =
    scrollAxis === "vertical"
      ? "flex min-h-full items-start justify-center overflow-visible p-0"
      : "flex min-h-full min-w-max items-center justify-start overflow-visible px-8 py-6";
  const previewTransformClass =
    scrollAxis === "vertical" ? "flex w-full flex-col items-center gap-0 leading-none" : "will-change-transform";
  const previewImageClass =
    scrollAxis === "vertical"
      ? "w-auto max-w-full select-none rounded-2xl border border-border object-contain shadow-[0_20px_60px_rgba(0,0,0,0.55)]"
      : "h-auto max-h-[calc(100vh-180px)] w-auto max-w-none select-none rounded-2xl border border-border object-contain shadow-[0_20px_60px_rgba(0,0,0,0.55)]";
  const scrollHint =
    scrollAxis === "vertical"
      ? "Scroll: rolar vertical • Ctrl+scroll: zoom • Space+drag: mover"
      : "Scroll: rolar horizontal • Ctrl+scroll: zoom • Space+drag: mover";
  const previewTransformOrigin = scrollAxis === "vertical" ? "top center" : "center center";
  const previewTransform =
    scrollAxis === "vertical" ? "translate(0px, 0px)" : `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`;
  const readerImageWidth = `${Math.round(zoom * 10000) / 100}%`;
  const hasPreviewImage = scrollAxis === "vertical" ? readerImages.length > 0 : Boolean(page && imageSrc);
  const hasBatchReturn = Boolean(batchCompletion);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-bg-secondary px-6 py-3">
        <div className="flex items-center gap-3">
          <button
            data-testid={hasBatchReturn ? "preview-return-batch" : undefined}
            onClick={() => navigate(hasBatchReturn ? "/processing" : "/")}
            title={hasBatchReturn ? "Voltar para o lote concluido" : "Voltar para o inicio"}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-text-secondary transition-smooth hover:bg-bg-tertiary hover:text-text-primary"
          >
            <ChevronLeft size={18} />
            {hasBatchReturn && <span className="text-xs font-medium">Lote</span>}
          </button>
          <div>
            <p className="text-sm font-medium">{project?.obra}</p>
            <p className="text-xs text-text-secondary">
              Capitulo {project?.capitulo} - {totalPages} paginas
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {staleRenderPages.length > 0 && (
            <span
              className="rounded-full border border-status-warning/25 bg-status-warning/10 px-2.5 py-1 text-[11px] text-status-warning"
              title="Existem paginas com preview final desatualizado."
            >
              Preview final pendente
            </span>
          )}

          <button
            onClick={() => setShowOriginal(!showOriginal)}
            title={getPreviewToggleLabel(showOriginal)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs transition-smooth ${
              showOriginal
                ? "border border-status-warning/20 bg-status-warning/10 text-status-warning"
                : "border border-border bg-bg-tertiary text-text-secondary"
            }`}
          >
            {showOriginal ? <EyeOff size={14} /> : <Eye size={14} />}
            {getPreviewToggleLabel(showOriginal)}
          </button>

          <button
            onClick={() => navigate("/editor")}
            className="flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white transition-smooth hover:bg-brand-600"
          >
            Abrir Editor
          </button>

          <button
            data-testid="export-panel-toggle"
            onClick={() => setShowExportPanel(!showExportPanel)}
            className="flex items-center gap-1.5 rounded-lg bg-brand/10 px-3 py-1.5 text-xs text-brand-300 transition-smooth hover:bg-brand/20"
          >
            <Download size={14} />
            Exportar
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(72,176,255,0.08),_transparent_38%),linear-gradient(180deg,_rgba(255,255,255,0.02),_transparent_28%)]">
          {hasPreviewImage ? (
            <div
              data-testid="preview-zoom-toolbar"
              className="absolute right-4 top-4 z-30 flex items-center gap-1.5 rounded-xl border border-border bg-bg-secondary/90 px-2 py-2 shadow-lg backdrop-blur"
            >
              <button
                onClick={() => setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "out"))}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Diminuir zoom (-)"
              >
                <Minus size={14} />
              </button>
              <button
                onClick={() => setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "in"))}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Aumentar zoom (+)"
              >
                <Plus size={14} />
              </button>
              <button
                onClick={resetPreviewView}
                className="rounded-xl bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                title="Resetar zoom e posicao (0)"
              >
                Ajustar
              </button>
              <button
                onClick={() => setZoom(2)}
                className="rounded-xl bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                title="Zoom 2x"
              >
                2x
              </button>
              <button
                onClick={() => setPanOffset({ x: 0, y: 0 })}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Centralizar (pan)"
              >
                <LocateFixed size={14} />
              </button>
              <span data-testid="preview-zoom-value" className="w-12 text-right font-mono text-[11px] text-text-muted">
                {Math.round(zoom * 100)}%
              </span>
              <div className="ml-1 flex rounded-xl border border-border bg-bg-primary/70 p-0.5">
                <button
                  onClick={() => setScrollAxis("vertical")}
                  aria-pressed={scrollAxis === "vertical"}
                  className={`rounded-lg px-2 py-1 text-[11px] transition-smooth ${
                    scrollAxis === "vertical"
                      ? "bg-brand/20 text-brand-200"
                      : "text-text-secondary hover:text-text-primary"
                  }`}
                  title="Rolagem vertical"
                >
                  Vertical
                </button>
                <button
                  onClick={() => setScrollAxis("horizontal")}
                  aria-pressed={scrollAxis === "horizontal"}
                  className={`rounded-lg px-2 py-1 text-[11px] transition-smooth ${
                    scrollAxis === "horizontal"
                      ? "bg-brand/20 text-brand-200"
                      : "text-text-secondary hover:text-text-primary"
                  }`}
                  title="Rolagem horizontal"
                >
                  Horizontal
                </button>
              </div>
            </div>
          ) : null}

          {scrollAxis === "horizontal" ? (
            <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center px-4">
              <div className="rounded-full border border-border bg-black/45 px-3 py-1 text-[11px] text-text-secondary backdrop-blur">
                {scrollHint}
              </div>
            </div>
          ) : null}

          <div
            ref={viewportRef}
            data-testid="preview-viewport"
            className={`h-full ${viewportOverflowClass} ${viewportCursor}`}
            onScroll={handleViewportScroll}
            onWheel={handleViewportWheel}
            onMouseDown={handleViewportMouseDown}
          >
            <div className={previewFrameClass}>
              <div
                className={previewTransformClass}
                style={{
                  transform: previewTransform,
                  transformOrigin: previewTransformOrigin,
                  transition: panSession ? "none" : "transform 0.12s ease-out",
                }}
              >
                {scrollAxis === "vertical" ? (
                  readerImages.length > 0 ? (
                    readerImages.map((item) => (
                      <div
                        key={item.pageIndex}
                        ref={(node) => {
                          readerPageRefs.current[item.pageIndex] = node;
                        }}
                        className="m-0 flex w-full min-w-0 justify-center p-0 leading-none"
                      >
                        {item.src ? (
                          <img
                            src={item.src}
                            alt={`Pagina ${item.pageNumber}`}
                            draggable={false}
                            className="block h-auto max-w-none select-none"
                            style={{ width: readerImageWidth }}
                          />
                        ) : (
                          <div
                            className="flex h-64 items-center justify-center text-sm text-text-secondary"
                            style={{ width: readerImageWidth }}
                          >
                            {item.status === "error" ? "Imagem indisponivel" : "Carregando imagem..."}
                          </div>
                        )}
                      </div>
                    ))
                  ) : page ? (
                    <p className="px-6 py-8 text-sm text-text-secondary">Carregando imagens...</p>
                  ) : (
                    <p className="px-6 py-8 text-text-secondary">Nenhuma pagina para exibir</p>
                  )
                ) : page && imageSrc ? (
                  <div className="relative inline-block">
                    <img
                      ref={imgRef}
                      src={imageSrc}
                      alt={`Pagina ${page.numero}`}
                      draggable={false}
                      className={previewImageClass}
                    />
                  </div>
                ) : page ? (
                  <p className="text-sm text-text-secondary">Carregando imagem...</p>
                ) : (
                  <p className="text-text-secondary">Nenhuma pagina para exibir</p>
                )}
              </div>
            </div>
          </div>
        </div>

        <aside data-testid="qa-panel" className="w-80 overflow-y-auto border-l border-border bg-bg-secondary p-5">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-medium">Relatorio do capitulo</h3>
              <p className="mt-1 text-xs text-text-secondary">Revisao profissional antes do export.</p>
            </div>
            <span
              data-testid="qa-issue-count"
              className={`rounded-full px-2 py-1 font-mono text-xs ${
                qaIssues.length > 0
                  ? "bg-status-warning/10 text-status-warning"
                  : "bg-status-success/10 text-status-success"
              }`}
            >
              {qaIssues.length}
            </span>
          </div>

          <div data-testid="qa-review-report" className="mb-4 rounded-xl border border-border bg-bg-tertiary/70 p-3">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <p className="text-text-muted">Paginas</p>
                <p className="text-sm font-medium text-text-primary">{qaReviewSummary.totalPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Aprovadas</p>
                <p className="text-sm font-medium text-status-success">{qaReviewSummary.approvedPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Com aviso</p>
                <p className="text-sm font-medium text-status-warning">{qaReviewSummary.warningPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Bloqueadas</p>
                <p data-testid="qa-blocked-pages" className="text-sm font-medium text-status-error">{qaReviewSummary.blockedPages}</p>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 border-t border-border pt-3 text-xs">
              <div>
                <p className="text-text-muted">Criticos</p>
                <p data-testid="qa-critical-count" className="text-sm font-medium text-status-error">{qaReviewSummary.criticalCount}</p>
              </div>
              <div>
                <p className="text-text-muted">Warnings</p>
                <p className="text-sm font-medium text-status-warning">{qaReviewSummary.warningCount}</p>
              </div>
            </div>
          </div>

          <div data-testid="qa-group-list" className="mb-4 space-y-1">
            {qaGroups.length === 0 ? (
              <div className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary">
                Sem grupos ativos.
              </div>
            ) : (
              qaGroups.map(([group, count]) => (
                <div key={group} className="flex items-center justify-between rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                  <span className="text-text-secondary">{group}</span>
                  <span className="font-mono text-text-primary">{count}</span>
                </div>
              ))
            )}
          </div>

          <div className="space-y-2">
            {qaIssues.length === 0 ? (
              <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
                Nenhuma flag ativa.
              </div>
            ) : (
              qaIssues.map((issue) => (
                <div key={issue.id} className="rounded-lg border border-border bg-bg-tertiary p-3">
                  <button
                    data-testid="qa-flag-item"
                    onClick={() => goToQaIssue(issue)}
                    className="flex w-full items-start gap-2 text-left"
                  >
                    <AlertTriangle
                      size={16}
                      className={
                        issue.severity === "critical" || issue.severity === "high"
                          ? "mt-0.5 text-status-error"
                          : "mt-0.5 text-status-warning"
                      }
                    />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-medium text-text-primary">{issue.label}</span>
                        <span className="mt-1 inline-flex rounded-md bg-bg-primary px-2 py-0.5 text-[10px] text-text-muted">
                          {qaIssueGroup(issue.flagId)}
                        </span>
                        <span className="mt-1 block text-xs text-text-secondary">
                          Pagina {issue.pageNumber} - regiao {issue.regionId}
                      </span>
                      <span className="mt-1 block truncate text-[11px] text-text-muted">{issue.sourceText}</span>
                    </span>
                  </button>

                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <button
                      onClick={() => goToQaIssue(issue)}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <LocateFixed size={12} />
                      Ir para pagina
                    </button>
                    <button
                      onClick={() => navigate("/editor")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <Edit3 size={12} />
                      Corrigir texto
                    </button>
                    <button
                      onClick={() => navigate("/setup")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <ShieldPlus size={12} />
                      Glossario
                    </button>
                    <button
                      onClick={() => alert("Regiao marcada para reprocessamento.")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <RotateCw size={12} />
                      Reprocessar
                    </button>
                    <button
                      onClick={() => alert("Mascara marcada para regeneracao.")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <RotateCw size={12} />
                      Mascara
                    </button>
                  </div>

                  {activeIgnoreIssue?.id === issue.id ? (
                    <div className="mt-3 space-y-2">
                      <textarea
                        data-testid="qa-ignore-reason"
                        value={ignoreReason}
                        onChange={(event) => {
                          setIgnoreReason(event.target.value);
                          setIgnoreError(null);
                        }}
                        placeholder="Motivo para ignorar"
                        className="min-h-[72px] w-full rounded-md border border-border bg-bg-primary px-2 py-2 text-xs text-text-primary outline-none focus:border-brand/50"
                      />
                      {ignoreError && <p className="text-xs text-status-error">{ignoreError}</p>}
                      <div className="flex gap-2">
                        <button
                          data-testid="qa-save-ignore"
                          onClick={confirmIgnoreIssue}
                          disabled={ignoreReason.trim().length === 0}
                          className="flex flex-1 items-center justify-center gap-1 rounded-md bg-brand px-2 py-1.5 text-[11px] font-medium text-white transition-smooth hover:bg-brand-600 disabled:opacity-40"
                        >
                          <Check size={12} />
                          Salvar motivo
                        </button>
                        <button
                          onClick={() => setIgnoreIssueId(null)}
                          className="rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                        >
                          Cancelar
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      data-testid="qa-ignore-button"
                      onClick={() => startIgnoreIssue(issue)}
                      className="mt-3 w-full rounded-md border border-status-warning/25 bg-status-warning/10 px-2 py-1.5 text-[11px] font-medium text-status-warning transition-smooth hover:bg-status-warning/15"
                    >
                      Ignorar com motivo
                    </button>
                  )}
                </div>
              ))
            )}
          </div>

          <button
            onClick={handleExportQaReport}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:bg-white/[0.03] hover:text-text-primary"
          >
            <FileText size={14} />
            Exportar relatorio
          </button>

          {(lastIgnoredReason || ignoredQaActions.length > 0) && (
            <div className="mt-4 rounded-lg border border-status-success/20 bg-status-success/10 p-3 text-xs text-status-success">
              {lastIgnoredReason ?? ignoredQaActions[ignoredQaActions.length - 1]?.ignored_reason}
            </div>
          )}
        </aside>

        {showExportPanel && (
          <div className="w-72 space-y-4 border-l border-border bg-bg-secondary p-5">
            <h3 className="text-sm font-medium">Exportar projeto</h3>

            <div className="space-y-2">
              {(
                [
                  { value: "zip_full", label: "ZIP completo", desc: "Originais + traduzidas + project.json" },
                  { value: "jpg_only", label: "Somente traduzidas", desc: "Apenas as imagens traduzidas" },
                  { value: "cbz", label: "CBZ", desc: "Formato de leitor de manga" },
                  { value: "psd", label: "Photoshop (PSD)", desc: "Camadas separadas: Original, Inpaint, Texto" },
                ] as const
              ).map((option) => (
                <button
                  key={option.value}
                  onClick={() => setExportFormat(option.value)}
                  className={`w-full rounded-lg border p-3 text-left transition-smooth ${
                    exportFormat === option.value
                      ? "border-brand/25 bg-brand/5"
                      : "border-border hover:border-border"
                  }`}
                >
                  <p className="text-sm font-medium">{option.label}</p>
                  <p className="mt-0.5 text-xs text-text-secondary">{option.desc}</p>
                </button>
              ))}
            </div>

            <div data-testid="export-mode-options" className="space-y-2">
              {EXPORT_MODE_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  data-testid={`export-mode-${option.id}`}
                  type="button"
                  onClick={() => setExportMode(option.id)}
                  className={`w-full rounded-lg border p-3 text-left transition-smooth ${
                    exportMode === option.id
                      ? "border-brand/25 bg-brand/5"
                      : "border-border hover:border-border"
                  }`}
                >
                  <p className="text-sm font-medium">{option.label}</p>
                  <p className="mt-0.5 text-xs text-text-secondary">{option.description}</p>
                </button>
              ))}
            </div>

            <div className="space-y-2 pt-2">
              {exportBlockMessage && (
                <div data-testid="export-block-message" className="rounded-lg border border-status-error/25 bg-status-error/10 px-3 py-2 text-xs text-status-error">
                  {exportBlockMessage}
                </div>
              )}
              <button
                data-testid="export-button"
                onClick={() => handleExport()}
                disabled={exporting}
                className="w-full rounded-lg bg-brand py-2.5 text-sm font-medium text-white transition-smooth hover:bg-brand-600 disabled:opacity-50"
              >
                {exporting ? "Exportando..." : "Exportar limpo"}
              </button>

              <button
                data-testid="export-with-warnings-button"
                onClick={() => handleExport({ mode: "with_warnings" })}
                disabled={exporting}
                className="w-full rounded-lg border border-status-warning/25 bg-status-warning/10 py-2 text-xs font-medium text-status-warning transition-smooth hover:bg-status-warning/15 disabled:opacity-50"
              >
                Exportar com avisos
              </button>

              <button
                data-testid="export-debug-button"
                onClick={() => handleExport({ mode: "debug" })}
                disabled={exporting}
                className="w-full rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-50"
              >
                Exportar debug
              </button>

              <button
                data-testid="export-report-link"
                onClick={handleExportLog}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:bg-white/[0.03] hover:text-text-primary"
              >
                <FileText size={14} />
                Exportar Log do Pipeline
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-4 border-t border-border bg-bg-secondary px-6 py-3">
        <button
          onClick={() => goToPreviewPage(currentPage - 1)}
          onFocus={() => preloadPreviewPage(currentPage - 1)}
          onMouseEnter={() => preloadPreviewPage(currentPage - 1)}
          disabled={currentPage === 0}
          title="Pagina anterior"
          className="rounded-lg bg-bg-tertiary p-2 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
        >
          <ArrowLeft size={16} />
        </button>

        <span data-testid="preview-page-counter" className="min-w-[80px] text-center font-mono text-sm text-text-secondary">
          {currentPage + 1} / {totalPages}
        </span>

        <button
          onClick={() => goToPreviewPage(currentPage + 1)}
          onFocus={() => preloadPreviewPage(currentPage + 1)}
          onMouseEnter={() => preloadPreviewPage(currentPage + 1)}
          disabled={currentPage >= totalPages - 1}
          title="Proxima pagina"
          className="rounded-lg bg-bg-tertiary p-2 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
        >
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
