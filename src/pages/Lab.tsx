import { useEffect, useState } from "react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useLocation, useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Bot,
  CheckCircle2,
  CheckSquare,
  ChevronRight,
  Clock3,
  FlaskConical,
  FolderOpen,
  FolderSearch,
  GitBranch,
  Pause,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Square,
  XCircle,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { useLabStore } from "../lib/stores/labStore";
import {
  type LabChapterPair,
  type LabChapterScopeMode,
  type LabCoderStrategy,
  type LabGpuPolicy,
  type LabPatchApplyResult,
  type LabPatchProposal,
  type StartLabRequest,
  applyLabPatch,
  approveLabBatch,
  approveLabProposal,
  getLabReferencePreview,
  getLabState,
  onLabAgentStatus,
  onLabBenchmarkResult,
  onLabProposalPromoted,
  onLabReviewRequested,
  onLabReviewResult,
  onLabState,
  openFiles,
  pauseLab,
  pickLabReferenceDir,
  pickLabSourceDir,
  proposeLabPatch,
  rejectLabProposal,
  resumeLab,
  setLabDirs,
  startLab,
  stopLab,
} from "../lib/tauri";

type LabSection = "home" | "run" | "reviews" | "decisions" | "benchmarks" | "history";

const SECTION_LABELS: Record<LabSection, string> = {
  home: "Visao geral",
  run: "Run atual",
  reviews: "Fila de revisao",
  decisions: "Decisoes",
  benchmarks: "Benchmarks",
  history: "Historico",
};

function extractErrorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  return "Falha inesperada no Lab.";
}

function formatEta(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remaining = safe % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${remaining}s`;
  return `${remaining}s`;
}

function formatTimestamp(timestampMs: number): string {
  if (!timestampMs) return "ainda nao registrado";
  return new Date(timestampMs).toLocaleString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  });
}

function toneForStatus(status: string): string {
  if (status === "running" || status === "approve" || status === "approved") {
    return "border-status-success/25 bg-status-success/10 text-status-success";
  }
  if (status === "paused" || status === "needs_benchmark_focus" || status === "benchmark_passed") {
    return "border-status-warning/25 bg-status-warning/10 text-status-warning";
  }
  if (status === "error" || status === "block" || status === "rejected" || status === "benchmark_failed") {
    return "border-status-error/25 bg-status-error/10 text-status-error";
  }
  return "border-white/10 bg-white/5 text-text-secondary";
}

function labelForVerdict(verdict: string): string {
  switch (verdict) {
    case "approve": return "Aprova";
    case "request_changes": return "Pede ajustes";
    case "block": return "Bloqueia";
    case "needs_benchmark_focus": return "Pede foco em benchmark";
    case "pending": return "Pendente";
    default: return verdict || "Pendente";
  }
}

function labelForProposalStatus(status: string): string {
  switch (status) {
    case "reviewing": return "Em revisao";
    case "benchmark_passed": return "Benchmark verde";
    case "benchmark_failed": return "Benchmark falhou";
    case "approved": return "Aprovada";
    case "rejected": return "Rejeitada";
    default: return status || "Aguardando";
  }
}

function compactReviewerLabel(reviewerId: string): string {
  switch (reviewerId) {
    case "python_senior_reviewer": return "Python";
    case "rust_senior_reviewer": return "Rust";
    case "react_ts_senior_reviewer": return "React/TS";
    case "tauri_boundary_reviewer": return "Tauri boundary";
    case "integration_architect": return "Integrador";
    default: return reviewerId;
  }
}

function labelForGpuPolicy(policy: LabGpuPolicy | string): string {
  return policy === "require_gpu" ? "GPU estrita" : "GPU preferencial";
}

function describeSelectedScope(selectedPairs: LabChapterPair[], totalPairs: number): string {
  if (selectedPairs.length === 0) return "Nenhum capitulo selecionado";
  if (selectedPairs.length === totalPairs) return "Todos os capitulos";
  if (selectedPairs.length === 1) return `Capitulo ${selectedPairs[0].chapter_number}`;
  const numbers = selectedPairs.map((pair) => pair.chapter_number);
  const contiguous = numbers.every((chapter, index) => chapter === numbers[0] + index);
  if (contiguous) return `Capitulos ${numbers[0]}-${numbers[numbers.length - 1]}`;
  return `${selectedPairs.length} capitulos selecionados`;
}

function useObjectUrl(filePath: string | null) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!filePath) {
      setImageUrl(null);
      setError(null);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;

    readFile(filePath)
      .then((bytes) => {
        if (cancelled) return;
        const extension = filePath.split(".").pop()?.toLowerCase();
        const mime = extension === "png" ? "image/png" : extension === "webp" ? "image/webp" : "image/jpeg";
        objectUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
        setImageUrl(objectUrl);
        setError(null);
      })
      .catch((readError) => {
        if (cancelled) return;
        setImageUrl(null);
        setError(extractErrorMessage(readError));
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [filePath]);

  return { imageUrl, error };
}

export function Lab() {
  const navigate = useNavigate();
  const location = useLocation();
  const systemProfile = useAppStore((state) => state.systemProfile);

  const snapshot = useLabStore((state) => state.snapshot);
  const referencePreview = useLabStore((state) => state.referencePreview);
  const previewLoading = useLabStore((state) => state.previewLoading);
  const previewError = useLabStore((state) => state.previewError);
  const selectedChapter = useLabStore((state) => state.selectedChapter);
  const selectedPage = useLabStore((state) => state.selectedPage);
  const highlightedProposalId = useLabStore((state) => state.highlightedProposalId);
  const setSnapshot = useLabStore((state) => state.setSnapshot);
  const setReferencePreview = useLabStore((state) => state.setReferencePreview);
  const setPreviewLoading = useLabStore((state) => state.setPreviewLoading);
  const setPreviewError = useLabStore((state) => state.setPreviewError);
  const setSelectedChapter = useLabStore((state) => state.setSelectedChapter);
  const setSelectedPage = useLabStore((state) => state.setSelectedPage);
  const setHighlightedProposalId = useLabStore((state) => state.setHighlightedProposalId);

  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);
  const [liveSignal, setLiveSignal] = useState("Conectando a telemetria do Lab...");
  const [chapterScopeMode, setChapterScopeMode] = useState<LabChapterScopeMode>("all");
  const [firstChapterCount, setFirstChapterCount] = useState("15");
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const [rangeEnd, setRangeEnd] = useState<number | null>(null);
  const [gpuPolicy, setGpuPolicy] = useState<LabGpuPolicy>("require_gpu");
  const [selectedLabFile, setSelectedLabFile] = useState<string | null>(null);
  const [explicitChapters, setExplicitChapters] = useState<Set<number>>(new Set());
  const [chapterFilter, setChapterFilter] = useState("");
  const [patchModal, setPatchModal] = useState<{ proposalId: string; patch: LabPatchProposal } | null>(null);
  const [patchCoderStrategy, setPatchCoderStrategy] = useState<LabCoderStrategy>("local");
  const [applyResult, setApplyResult] = useState<LabPatchApplyResult | null>(null);
  const [applyConfirm, setApplyConfirm] = useState(false); // aguardando confirmacao

  const pathParts = location.pathname.split("/").filter(Boolean);
  const allowedSections: LabSection[] = ["home", "run", "reviews", "decisions", "benchmarks", "history"];
  const sectionCandidate = pathParts[1] ?? "home";
  const activeSection: LabSection = allowedSections.includes(sectionCandidate as LabSection)
    ? (sectionCandidate as LabSection)
    : "home";
  const routeTargetId = pathParts[2] ? decodeURIComponent(pathParts[2]) : null;

  const catalogChapterPairs = snapshot?.available_chapter_pairs?.length
    ? snapshot.available_chapter_pairs
    : snapshot?.chapter_pairs ?? [];
  const runChapterPairs = snapshot?.chapter_pairs?.length
    ? snapshot.chapter_pairs
    : catalogChapterPairs;
  const availableChapterNumbers = catalogChapterPairs.map((pair) => pair.chapter_number);
  const selectedPair = runChapterPairs.find((pair) => pair.chapter_number === selectedChapter)
    ?? catalogChapterPairs.find((pair) => pair.chapter_number === selectedChapter)
    ?? snapshot?.chapter_pairs[0]
    ?? catalogChapterPairs[0]
    ?? null;
  const parsedFirstChapterCount = Math.max(1, Number.parseInt(firstChapterCount, 10) || 1);
  const effectiveRangeStart = rangeStart ?? availableChapterNumbers[0] ?? null;
  const effectiveRangeEnd = rangeEnd ?? availableChapterNumbers[availableChapterNumbers.length - 1] ?? null;
  const scopedChapterPairs = (() => {
    if (chapterScopeMode === "first_n") {
      return catalogChapterPairs.slice(0, parsedFirstChapterCount);
    }
    if (chapterScopeMode === "range" && effectiveRangeStart !== null && effectiveRangeEnd !== null) {
      const start = Math.min(effectiveRangeStart, effectiveRangeEnd);
      const end = Math.max(effectiveRangeStart, effectiveRangeEnd);
      return catalogChapterPairs.filter((pair) => pair.chapter_number >= start && pair.chapter_number <= end);
    }
    if (chapterScopeMode === "explicit") {
      return catalogChapterPairs.filter((pair) => explicitChapters.has(pair.chapter_number));
    }
    return catalogChapterPairs;
  })();

  const filteredCatalogPairs = chapterFilter.trim()
    ? catalogChapterPairs.filter((pair) =>
        String(pair.chapter_number).includes(chapterFilter.trim())
        || pair.reference_group.toLowerCase().includes(chapterFilter.trim().toLowerCase())
      )
    : catalogChapterPairs;
  const scopeSummary = describeSelectedScope(scopedChapterPairs, catalogChapterPairs.length);
  const maxPageIndex = Math.max(0, Math.max(selectedPair?.source_pages ?? 1, selectedPair?.reference_pages ?? 1) - 1);
  const selectedProposal = snapshot?.proposals.find((proposal) => proposal.proposal_id === highlightedProposalId)
    ?? snapshot?.proposals[0]
    ?? null;
  const selectedReviews = snapshot?.reviews.filter((review) => review.proposal_id === selectedProposal?.proposal_id) ?? [];
  const outputImage = useObjectUrl(referencePreview?.output_path ?? null);
  const referenceImage = useObjectUrl(referencePreview?.reference_path ?? null);

  useEffect(() => {
    if (availableChapterNumbers.length === 0) {
      setRangeStart(null);
      setRangeEnd(null);
      return;
    }

    setRangeStart((current) => (current !== null && availableChapterNumbers.includes(current) ? current : availableChapterNumbers[0]));
    setRangeEnd((current) => (
      current !== null && availableChapterNumbers.includes(current)
        ? current
        : availableChapterNumbers[availableChapterNumbers.length - 1]
    ));
  }, [snapshot?.available_chapter_pairs, snapshot?.chapter_pairs]);

  useEffect(() => {
    if (location.pathname === "/lab") {
      navigate("/lab/home", { replace: true });
      return;
    }
    if (sectionCandidate !== activeSection) {
      navigate("/lab/home", { replace: true });
    }
  }, [activeSection, location.pathname, navigate, sectionCandidate]);

  useEffect(() => {
    let disposed = false;
    let unlisteners: Array<() => void> = [];

    async function bootstrap() {
      try {
        const initial = await getLabState();
        if (disposed) return;
        setSnapshot(initial);
        setLiveSignal(`Lab pronto: ${initial.chapter_pairs.length} capitulos pareados.`);
      } catch (error) {
        if (!disposed) setPageError(extractErrorMessage(error));
      }

      try {
        const listeners = await Promise.all([
          onLabState((nextSnapshot) => {
            setSnapshot(nextSnapshot);
            setLiveSignal(`${nextSnapshot.current_stage || "estado"}: ${nextSnapshot.message}`);
          }),
          onLabAgentStatus((agent) => {
            setLiveSignal(`${agent.label}: ${agent.last_action || agent.current_task}`);
          }),
          onLabReviewRequested((proposal) => {
            setLiveSignal(`Revisao aberta para "${proposal.title}".`);
          }),
          onLabReviewResult((review) => {
            setLiveSignal(`${review.reviewer_label}: ${labelForVerdict(review.verdict)}.`);
          }),
          onLabBenchmarkResult((benchmark) => {
            setLiveSignal(
              benchmark.green
                ? `Benchmark verde para ${benchmark.proposal_id}.`
                : `Benchmark reprovou ${benchmark.proposal_id}.`
            );
          }),
          onLabProposalPromoted((event) => {
            setLiveSignal(event.summary);
          }),
        ]);

        if (disposed) {
          listeners.forEach((dispose) => dispose());
          return;
        }
        unlisteners = listeners;
      } catch (error) {
        if (!disposed) setPageError(extractErrorMessage(error));
      }
    }

    bootstrap();
    return () => {
      disposed = true;
      unlisteners.forEach((dispose) => dispose());
    };
  }, [setSnapshot]);

  useEffect(() => {
    const shouldPoll =
      snapshot?.status === "starting"
      || snapshot?.status === "running"
      || snapshot?.status === "stopping";
    if (!shouldPoll) {
      return;
    }

    const intervalId = window.setInterval(() => {
      getLabState()
        .then((nextSnapshot) => {
          setSnapshot(nextSnapshot);
        })
        .catch(() => {
          // Keep the current live state when a polling refresh fails.
        });
    }, 2000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [setSnapshot, snapshot?.status]);

  useEffect(() => {
    if (!snapshot?.proposals.length) return;

    if (activeSection === "reviews" && routeTargetId) {
      if (snapshot.proposals.some((proposal) => proposal.proposal_id === routeTargetId)) {
        setHighlightedProposalId(routeTargetId);
      } else {
        setHighlightedProposalId(snapshot.proposals[0]?.proposal_id ?? null);
      }
      return;
    }

    if (!highlightedProposalId) {
      setHighlightedProposalId(snapshot.proposals[0]?.proposal_id ?? null);
    }
  }, [activeSection, highlightedProposalId, routeTargetId, setHighlightedProposalId, snapshot?.proposals]);

  useEffect(() => {
    if (selectedPage > maxPageIndex) {
      setSelectedPage(0);
    }
  }, [maxPageIndex, selectedPage, setSelectedPage]);

  useEffect(() => {
    if (!selectedChapter) {
      setReferencePreview(null);
      setPreviewLoading(false);
      setPreviewError(null);
      return;
    }

    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError(null);

    getLabReferencePreview(selectedChapter, selectedPage)
      .then((preview) => {
        if (cancelled) return;
        setReferencePreview(preview);
        setPreviewLoading(false);
      })
      .catch((error) => {
        if (cancelled) return;
        setReferencePreview(null);
        setPreviewLoading(false);
        setPreviewError(extractErrorMessage(error));
      });

    return () => {
      cancelled = true;
    };
  }, [selectedChapter, selectedPage, setPreviewError, setPreviewLoading, setReferencePreview]);

  async function refreshSnapshot() {
    try {
      const nextSnapshot = await getLabState();
      setSnapshot(nextSnapshot);
      setPageError(null);
    } catch (error) {
      setPageError(extractErrorMessage(error));
    }
  }

  async function runAction(actionId: string, work: () => Promise<void>) {
    setBusyAction(actionId);
    setPageError(null);
    try {
      await work();
    } catch (error) {
      setPageError(extractErrorMessage(error));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartLab() {
    await runAction("start", async () => {
      let chapter_scope: StartLabRequest["chapter_scope"];
      if (chapterScopeMode === "first_n") {
        chapter_scope = { mode: "first_n", first_n: parsedFirstChapterCount };
      } else if (
        chapterScopeMode === "range"
        && effectiveRangeStart !== null
        && effectiveRangeEnd !== null
      ) {
        chapter_scope = {
          mode: "range",
          start_chapter: Math.min(effectiveRangeStart, effectiveRangeEnd),
          end_chapter: Math.max(effectiveRangeStart, effectiveRangeEnd),
        };
      } else if (chapterScopeMode === "explicit") {
        const numbers = Array.from(explicitChapters).sort((a, b) => a - b);
        if (numbers.length === 0) {
          throw new Error("Marque ao menos um capitulo antes de iniciar a rodada.");
        }
        chapter_scope = { mode: "explicit", chapter_numbers: numbers };
      } else {
        chapter_scope = { mode: "all" };
      }

      const request: StartLabRequest = {
        chapter_scope,
        gpu_policy: gpuPolicy,
      };

      const response = await startLab(request);
      navigate(`/lab/run/${response.run_id}`);
      await refreshSnapshot();
    });
  }

  async function handlePickSourceDir() {
    try {
      const picked = await pickLabSourceDir();
      if (!picked) return;
      const referenceDir = snapshot?.reference_dir ?? "";
      if (!referenceDir) {
        setPageError("Selecione tambem uma pasta de referencia PT-BR para pareamento.");
        return;
      }
      const next = await setLabDirs(picked, referenceDir);
      setSnapshot(next);
      setPageError(null);
    } catch (error) {
      setPageError(extractErrorMessage(error));
    }
  }

  async function handlePickReferenceDir() {
    try {
      const picked = await pickLabReferenceDir();
      if (!picked) return;
      const sourceDir = snapshot?.source_dir ?? "";
      if (!sourceDir) {
        setPageError("Selecione tambem uma pasta de origem EN para pareamento.");
        return;
      }
      const next = await setLabDirs(sourceDir, picked);
      setSnapshot(next);
      setPageError(null);
    } catch (error) {
      setPageError(extractErrorMessage(error));
    }
  }

  function toggleExplicitChapter(chapterNumber: number) {
    setExplicitChapters((current) => {
      const next = new Set(current);
      if (next.has(chapterNumber)) {
        next.delete(chapterNumber);
      } else {
        next.add(chapterNumber);
      }
      return next;
    });
  }

  function selectAllExplicitChapters() {
    setExplicitChapters(new Set(filteredCatalogPairs.map((pair) => pair.chapter_number)));
  }

  function clearExplicitChapters() {
    setExplicitChapters(new Set());
  }

  function invertExplicitChapters() {
    setExplicitChapters((current) => {
      const next = new Set<number>();
      for (const pair of filteredCatalogPairs) {
        if (!current.has(pair.chapter_number)) {
          next.add(pair.chapter_number);
        }
      }
      return next;
    });
  }

  async function handleOpenFileForLab() {
    try {
      const path = await openFiles();
      if (path) {
        setSelectedLabFile(path);
        setPageError(null);
      }
    } catch (error) {
      setPageError(extractErrorMessage(error));
    }
  }

  async function handlePauseLab() {
    await runAction("pause", async () => {
      await pauseLab();
      await refreshSnapshot();
    });
  }

  async function handleResumeLab() {
    await runAction("resume", async () => {
      await resumeLab();
      await refreshSnapshot();
    });
  }

  async function handleStopLab() {
    await runAction("stop", async () => {
      await stopLab();
      await refreshSnapshot();
    });
  }

  async function handleApproveProposal(proposalId: string) {
    await runAction(`approve:${proposalId}`, async () => {
      await approveLabProposal(proposalId);
      await refreshSnapshot();
    });
  }

  async function handleRejectProposal(proposalId: string) {
    await runAction(`reject:${proposalId}`, async () => {
      await rejectLabProposal(proposalId);
      await refreshSnapshot();
    });
  }

  async function handleGeneratePatch(proposalId: string) {
    await runAction(`patch:${proposalId}`, async () => {
      const patch = await proposeLabPatch(proposalId, patchCoderStrategy);
      setPatchModal({ proposalId, patch });
      setApplyResult(null);
      setApplyConfirm(false);
    });
  }

  async function handleApplyPatch(proposalId: string, diff: string, createBranch: boolean) {
    await runAction(`apply:${proposalId}`, async () => {
      const result = await applyLabPatch(
        proposalId,
        diff,
        createBranch,
        createBranch, // commit somente se criou branch
        `Lab: patch automatico para ${proposalId}\n\nRevisado e aprovado pelo usuario via TraduzAi Lab.`
      );
      setApplyResult(result);
      setApplyConfirm(false);
      if (result.applied) await refreshSnapshot();
    });
  }

  async function handleApproveBatch(batchId: string) {
    await runAction(`batch:${batchId}`, async () => {
      await approveLabBatch(batchId);
      await refreshSnapshot();
    });
  }

  const running = snapshot?.status === "running" || snapshot?.status === "starting";
  const paused = snapshot?.status === "paused";
  const stoppable = running || paused;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(96,165,250,0.10),_rgba(11,15,27,0.98)_45%),linear-gradient(180deg,rgba(16,19,34,1)_0%,rgba(10,12,22,1)_100%)]">
      <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-6">
        <section className="rounded-[28px] border border-white/6 bg-[radial-gradient(circle_at_top_left,_rgba(41,187,255,0.14),_rgba(17,19,33,0.98)_58%)] p-6 space-y-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/6 text-[11px] uppercase tracking-[0.24em] text-text-secondary">
                <FlaskConical size={13} />
                Improvement Lab
              </div>
              <h1 className="text-3xl font-semibold text-text-primary">Painel de agentes e governanca tecnica</h1>
              <p className="text-sm text-text-secondary max-w-3xl">
                Monitore o runtime, a esteira hierarquica de revisao e o benchmark do corpus
                EN versus PT-BR antes de qualquer promocao local.
              </p>
            </div>

            <div className="min-w-[260px] rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
              <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Sinal ao vivo</p>
              <p className="text-sm text-text-primary mt-2 leading-relaxed">{liveSignal}</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleOpenFileForLab}
              disabled={running || busyAction !== null}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan text-sm font-medium hover:bg-accent-cyan/15 transition-smooth disabled:opacity-40"
            >
              <FolderOpen size={15} />
              Abrir arquivo
            </button>
            <button
              onClick={handleStartLab}
              disabled={running || snapshot?.status === "stopping" || busyAction !== null || catalogChapterPairs.length === 0}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-accent-purple text-white text-sm font-medium hover:bg-accent-purple-dark transition-smooth disabled:opacity-40"
            >
              <Play size={15} />
              Iniciar Lab
            </button>
            <button
              onClick={handlePauseLab}
              disabled={!running || busyAction !== null}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-status-warning/25 bg-status-warning/10 text-status-warning text-sm font-medium hover:bg-status-warning/15 transition-smooth disabled:opacity-40"
            >
              <Pause size={15} />
              Pausar
            </button>
            <button
              onClick={handleResumeLab}
              disabled={!paused || busyAction !== null}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan text-sm font-medium hover:bg-accent-cyan/15 transition-smooth disabled:opacity-40"
            >
              <Play size={15} />
              Retomar
            </button>
            <button
              onClick={handleStopLab}
              disabled={!stoppable || busyAction !== null}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-status-error/25 bg-status-error/10 text-status-error text-sm font-medium hover:bg-status-error/15 transition-smooth disabled:opacity-40"
            >
              <Square size={15} />
              Encerrar
            </button>
            <button
              onClick={() => refreshSnapshot()}
              disabled={busyAction !== null}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-white/10 bg-white/5 text-text-secondary text-sm font-medium hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-40"
            >
              <RefreshCw size={15} />
              Atualizar
            </button>
            {snapshot?.active_batch_id && (
              <button
                onClick={() => handleApproveBatch(snapshot.active_batch_id)}
                disabled={busyAction !== null}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-2xl border border-status-success/25 bg-status-success/10 text-status-success text-sm font-medium hover:bg-status-success/15 transition-smooth disabled:opacity-40"
              >
                <ShieldCheck size={15} />
                Aprovar lote
              </button>
            )}
          </div>

          <div className="rounded-2xl border border-white/8 bg-black/20 p-4 space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Fontes do corpus</p>
                <p className="text-sm text-text-secondary mt-1">
                  Aponte a pasta com CBZs EN e a pasta com referencia PT-BR. O pareamento e feito
                  por numero de capitulo.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handlePickSourceDir}
                  disabled={running || busyAction !== null}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan text-xs font-medium hover:bg-accent-cyan/15 transition-smooth disabled:opacity-40"
                >
                  <FolderSearch size={14} />
                  Selecionar pasta EN
                </button>
                <button
                  onClick={handlePickReferenceDir}
                  disabled={running || busyAction !== null}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border border-accent-purple/25 bg-accent-purple/10 text-accent-purple text-xs font-medium hover:bg-accent-purple/15 transition-smooth disabled:opacity-40"
                >
                  <FolderSearch size={14} />
                  Selecionar pasta PT-BR
                </button>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-3 text-xs text-text-secondary">
              <div className="rounded-xl border border-white/6 bg-black/30 px-3 py-2">
                <p className="uppercase tracking-[0.18em] text-[10px] text-text-secondary">EN (origem)</p>
                <p className="text-text-primary break-all mt-1">{snapshot?.source_dir || "Nao definido"}</p>
              </div>
              <div className="rounded-xl border border-white/6 bg-black/30 px-3 py-2">
                <p className="uppercase tracking-[0.18em] text-[10px] text-text-secondary">PT-BR (referencia)</p>
                <p className="text-text-primary break-all mt-1">{snapshot?.reference_dir || "Nao definido"}</p>
              </div>
            </div>
          </div>

          <div className="grid xl:grid-cols-[1.1fr_0.9fr] gap-4">
            <div className="rounded-2xl border border-white/8 bg-black/20 p-4 space-y-4">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Escopo da rodada</p>
                <p className="text-sm text-text-primary mt-2">
                  {scopeSummary} de {catalogChapterPairs.length} capitulos disponiveis.
                </p>
              </div>

              <div className="grid md:grid-cols-4 gap-2">
                {([
                  ["all", "Todos"],
                  ["first_n", "Primeiros N"],
                  ["range", "Intervalo"],
                  ["explicit", "Escolher"],
                ] as Array<[LabChapterScopeMode, string]>).map(([mode, label]) => (
                  <button
                    key={mode}
                    onClick={() => setChapterScopeMode(mode)}
                    disabled={running || busyAction !== null}
                    className={`px-3 py-2 rounded-2xl border text-sm transition-smooth ${
                      chapterScopeMode === mode
                        ? "border-accent-cyan/35 bg-accent-cyan/10 text-accent-cyan"
                        : "border-white/8 bg-white/4 text-text-secondary hover:text-text-primary hover:bg-white/6"
                    } disabled:opacity-40`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {chapterScopeMode === "first_n" && (
                <label className="block text-xs text-text-secondary">
                  Quantos capitulos iniciais entram na rodada
                  <input
                    type="number"
                    min={1}
                    max={Math.max(1, catalogChapterPairs.length)}
                    value={firstChapterCount}
                    onChange={(event) => setFirstChapterCount(event.target.value)}
                    disabled={running || busyAction !== null}
                    className="mt-2 block w-full rounded-xl bg-bg-secondary border border-white/8 px-3 py-2 text-sm text-text-primary"
                  />
                </label>
              )}

              {chapterScopeMode === "range" && (
                <div className="grid sm:grid-cols-2 gap-3">
                  <label className="block text-xs text-text-secondary">
                    Capitulo inicial
                    <select
                      value={effectiveRangeStart ?? ""}
                      onChange={(event) => setRangeStart(Number(event.target.value))}
                      disabled={running || busyAction !== null}
                      className="mt-2 block w-full rounded-xl bg-bg-secondary border border-white/8 px-3 py-2 text-sm text-text-primary"
                    >
                      {catalogChapterPairs.map((pair) => (
                        <option key={`start-${pair.chapter_number}`} value={pair.chapter_number}>
                          Cap. {pair.chapter_number}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-xs text-text-secondary">
                    Capitulo final
                    <select
                      value={effectiveRangeEnd ?? ""}
                      onChange={(event) => setRangeEnd(Number(event.target.value))}
                      disabled={running || busyAction !== null}
                      className="mt-2 block w-full rounded-xl bg-bg-secondary border border-white/8 px-3 py-2 text-sm text-text-primary"
                    >
                      {catalogChapterPairs.map((pair) => (
                        <option key={`end-${pair.chapter_number}`} value={pair.chapter_number}>
                          Cap. {pair.chapter_number}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              )}

              {chapterScopeMode === "explicit" && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative flex-1 min-w-[180px]">
                      <Search
                        size={14}
                        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary pointer-events-none"
                      />
                      <input
                        type="text"
                        value={chapterFilter}
                        onChange={(event) => setChapterFilter(event.target.value)}
                        placeholder="Buscar por numero ou grupo"
                        disabled={running || busyAction !== null}
                        className="block w-full rounded-xl bg-bg-secondary border border-white/8 pl-9 pr-3 py-2 text-sm text-text-primary"
                      />
                    </div>
                    <button
                      onClick={selectAllExplicitChapters}
                      disabled={running || busyAction !== null}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/8 bg-white/5 text-xs text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-40"
                    >
                      <CheckSquare size={13} />
                      Todos
                    </button>
                    <button
                      onClick={clearExplicitChapters}
                      disabled={running || busyAction !== null}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/8 bg-white/5 text-xs text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-40"
                    >
                      Nenhum
                    </button>
                    <button
                      onClick={invertExplicitChapters}
                      disabled={running || busyAction !== null}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/8 bg-white/5 text-xs text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-40"
                    >
                      Inverter
                    </button>
                  </div>

                  <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-1.5 max-h-[240px] overflow-y-auto rounded-xl border border-white/6 bg-black/25 p-2">
                    {filteredCatalogPairs.length === 0 ? (
                      <p className="col-span-full text-center text-xs text-text-secondary py-3">
                        Nenhum capitulo corresponde ao filtro atual.
                      </p>
                    ) : (
                      filteredCatalogPairs.map((pair) => {
                        const selected = explicitChapters.has(pair.chapter_number);
                        return (
                          <button
                            key={`explicit-${pair.chapter_number}`}
                            onClick={() => toggleExplicitChapter(pair.chapter_number)}
                            disabled={running || busyAction !== null}
                            title={`${pair.reference_group} - ${pair.source_pages} pag.`}
                            className={`px-2 py-1.5 rounded-lg text-xs border transition-smooth ${
                              selected
                                ? "border-accent-cyan/45 bg-accent-cyan/15 text-accent-cyan"
                                : "border-white/8 bg-white/4 text-text-secondary hover:text-text-primary hover:bg-white/8"
                            } disabled:opacity-40`}
                          >
                            {pair.chapter_number}
                          </button>
                        );
                      })
                    )}
                  </div>

                  <p className="text-xs text-text-secondary">
                    {explicitChapters.size} capitulo(s) marcado(s). Arbitrarios, fora de ordem, OK.
                  </p>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-white/8 bg-black/20 p-4 space-y-4">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Modo GPU do Lab</p>
                <p className="text-sm text-text-primary mt-2">
                  {labelForGpuPolicy(gpuPolicy)} para OCR, traducao local, detector visual e inpainting.
                </p>
                <p className="text-xs text-text-secondary mt-2">
                  O typesetting final continua em CPU por causa do renderer atual com Pillow.
                </p>
              </div>

              <div className="grid sm:grid-cols-2 gap-2">
                {([
                  ["require_gpu", "GPU estrita"],
                  ["prefer_gpu", "GPU preferencial"],
                ] as Array<[LabGpuPolicy, string]>).map(([policy, label]) => (
                  <button
                    key={policy}
                    onClick={() => setGpuPolicy(policy)}
                    disabled={running || busyAction !== null}
                    className={`px-3 py-2 rounded-2xl border text-sm transition-smooth ${
                      gpuPolicy === policy
                        ? "border-status-success/35 bg-status-success/10 text-status-success"
                        : "border-white/8 bg-white/4 text-text-secondary hover:text-text-primary hover:bg-white/6"
                    } disabled:opacity-40`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {pageError && (
            <div className="rounded-2xl border border-status-error/25 bg-status-error/10 px-4 py-3 text-sm text-status-error">
              {pageError}
            </div>
          )}

          {!snapshot?.git_available && (
            <div className="rounded-2xl border border-status-warning/25 bg-status-warning/10 px-4 py-3 text-sm text-status-warning">
              O workspace atual nao esta em um repositorio Git. O Lab continua executando, mas a etapa
              de PR local fica bloqueada ate existir um `.git`.
            </div>
          )}
        </section>

        <section className="grid sm:grid-cols-2 xl:grid-cols-5 gap-4">
          <div className="rounded-3xl border border-white/6 bg-white/4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Estado</p>
                <p className="text-xl font-semibold text-text-primary mt-2">{snapshot?.status ?? "carregando"}</p>
              </div>
              <Bot size={20} className="text-accent-purple" />
            </div>
            <p className="text-sm text-text-secondary mt-3">{snapshot?.message ?? "Lendo estado inicial do laboratorio..."}</p>
          </div>

          <div className="rounded-3xl border border-white/6 bg-white/4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Andamento</p>
                <p className="text-xl font-semibold text-text-primary mt-2">{snapshot?.processed_pairs ?? 0}/{snapshot?.total_pairs ?? 0}</p>
              </div>
              <Clock3 size={20} className="text-accent-cyan" />
            </div>
            <p className="text-sm text-text-secondary mt-3">ETA estimado: {formatEta(snapshot?.eta_seconds ?? 0)}</p>
            <p className="text-xs text-text-secondary mt-2">{snapshot?.scope_label || "Escopo ainda nao definido"}</p>
          </div>

          <div className="rounded-3xl border border-white/6 bg-white/4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Governanca</p>
                <p className="text-xl font-semibold text-text-primary mt-2">{snapshot?.pending_proposals ?? 0}</p>
              </div>
              <ShieldCheck size={20} className="text-status-success" />
            </div>
            <p className="text-sm text-text-secondary mt-3">Propostas aguardando revisao, benchmark ou sua decisao.</p>
          </div>

          <div className="rounded-3xl border border-white/6 bg-white/4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Infra local</p>
                <p className="text-lg font-semibold text-text-primary mt-2">{systemProfile?.gpu_name ?? "Hardware local"}</p>
              </div>
              <GitBranch size={20} className="text-status-warning" />
            </div>
            <p className="text-sm text-text-secondary mt-3">
              {snapshot?.acceleration_summary || "Aceleracao do pipeline ainda nao informada."}
            </p>
            <p className="text-xs text-text-secondary mt-2">
              {labelForGpuPolicy(snapshot?.gpu_policy || gpuPolicy)} | {snapshot?.git_available ? "Git detectado" : "Sem repositorio Git"} | Batch {snapshot?.active_batch_id || "nao iniciado"}
            </p>
          </div>

          <div className="rounded-3xl border border-white/6 bg-white/4 p-5">
            <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">PR local</p>
            <div className="mt-4 flex items-start gap-3">
              {snapshot?.pr_ready ? (
                <CheckCircle2 size={18} className="text-status-success mt-0.5" />
              ) : snapshot?.git_available ? (
                <AlertTriangle size={18} className="text-status-warning mt-0.5" />
              ) : (
                <XCircle size={18} className="text-status-error mt-0.5" />
              )}
              <div>
                <p className="text-sm font-medium text-text-primary">
                  {snapshot?.pr_ready ? "Ha proposta pronta" : snapshot?.git_available ? "Aguardando sua aprovacao" : "Bloqueada"}
                </p>
                <p className="text-xs text-text-secondary mt-2">Nada sobe sem benchmark verde e sem sua aprovacao manual.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-white/6 bg-white/4 p-3">
          <div className="grid md:grid-cols-3 xl:grid-cols-6 gap-2">
            {(["home", "run", "reviews", "decisions", "benchmarks", "history"] as LabSection[]).map((section) => (
              <button
                key={section}
                onClick={() => {
                  if (section === "run" && snapshot?.run_id) {
                    navigate(`/lab/run/${snapshot.run_id}`);
                    return;
                  }
                  if (section === "reviews" && selectedProposal?.proposal_id) {
                    navigate(`/lab/reviews/${selectedProposal.proposal_id}`);
                    return;
                  }
                  navigate(`/lab/${section}`);
                }}
                className={`px-4 py-3 rounded-2xl text-sm font-medium transition-smooth ${
                  activeSection === section
                    ? "bg-accent-purple/14 text-accent-purple border border-accent-purple/25"
                    : "bg-transparent text-text-secondary border border-transparent hover:bg-white/5 hover:text-text-primary"
                }`}
              >
                {SECTION_LABELS[section]}
              </button>
            ))}
          </div>
        </section>

        {(activeSection === "home" || activeSection === "run") && (
          <section className="grid xl:grid-cols-[1.1fr_0.9fr] gap-6">
            {selectedLabFile && (
              <div className="rounded-2xl border border-accent-cyan/25 bg-accent-cyan/10 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.24em] text-accent-cyan">Arquivo selecionado para Lab</p>
                <p className="text-sm text-text-primary mt-2 break-all">{selectedLabFile}</p>
              </div>
            )}
            <div className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Agentes</p>
                  <h2 className="text-lg font-semibold text-text-primary mt-1">Runtime e Improvement Lab</h2>
                </div>
                <span className="text-xs text-text-secondary">{snapshot?.agents.length ?? 0} agentes</span>
              </div>
              <div className="grid md:grid-cols-2 gap-4">
                {(snapshot?.agents ?? []).map((agent) => (
                  <div key={agent.agent_id} className="rounded-2xl border border-white/6 bg-black/20 p-4 space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium text-text-primary">{agent.label}</p>
                        <p className="text-xs text-text-secondary mt-1">{agent.layer === "review" ? "Esteira de revisao" : "Lab interno"}</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(agent.status)}`}>
                        {agent.status || "idle"}
                      </span>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">Tarefa atual</p>
                      <p className="text-sm text-text-primary mt-1">{agent.current_task || "Aguardando proximo item"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">Ultima acao</p>
                      <p className="text-sm text-text-primary mt-1">{agent.last_action || "Sem acao registrada"}</p>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-text-secondary">Confianca</span>
                      <span className="text-text-primary">{Math.round(agent.confidence * 100)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/6 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-accent-cyan to-accent-purple"
                        style={{ width: `${Math.max(0, Math.min(100, agent.confidence * 100))}%` }}
                      />
                    </div>
                    {agent.touched_domains.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {agent.touched_domains.map((domain) => (
                          <span key={`${agent.agent_id}-${domain}`} className="px-2 py-1 rounded-full bg-white/6 text-[11px] text-text-secondary">
                            {domain}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Referencia visual</p>
                  <h2 className="text-lg font-semibold text-text-primary mt-1">Saida do Lab versus scan PT-BR</h2>
                </div>
                <button
                  onClick={() => navigate("/lab/decisions")}
                  className="inline-flex items-center gap-2 text-xs text-accent-cyan hover:text-accent-purple transition-smooth"
                >
                  Abrir decisoes
                  <ChevronRight size={13} />
                </button>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-3">
                  <label className="text-xs text-text-secondary">
                    Capitulo
                    <select
                      value={selectedChapter ?? ""}
                      onChange={(event) => setSelectedChapter(Number(event.target.value))}
                      className="mt-1 block min-w-[180px] px-3 py-2 rounded-xl bg-bg-secondary border border-white/8 text-sm text-text-primary"
                    >
                      {runChapterPairs.map((pair) => (
                        <option key={pair.chapter_number} value={pair.chapter_number}>
                          Cap. {pair.chapter_number} - {pair.reference_group}
                        </option>
                      ))}
                    </select>
                  </label>

                  <div className="flex items-end gap-2">
                    <button
                      onClick={() => setSelectedPage(Math.max(0, selectedPage - 1))}
                      disabled={selectedPage <= 0}
                      className="h-[42px] px-3 rounded-xl border border-white/10 bg-white/5 text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-30"
                    >
                      <ArrowLeft size={16} />
                    </button>
                    <div className="px-4 py-2 rounded-xl border border-white/8 bg-bg-secondary text-sm text-text-primary">
                      Pagina {selectedPage + 1} / {maxPageIndex + 1}
                    </div>
                    <button
                      onClick={() => setSelectedPage(Math.min(maxPageIndex, selectedPage + 1))}
                      disabled={selectedPage >= maxPageIndex}
                      className="h-[42px] px-3 rounded-xl border border-white/10 bg-white/5 text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth disabled:opacity-30"
                    >
                      <ArrowRight size={16} />
                    </button>
                  </div>
                </div>

                <p className="text-xs text-text-secondary">
                  {previewLoading
                    ? "Atualizando preview..."
                    : previewError
                      ? previewError
                      : referencePreview?.output_kind === "source_fallback"
                        ? "Painel esquerdo ainda mostra o fallback EN."
                        : "Preview do output do Lab pronto."}
                </p>
              </div>

              <div className="grid xl:grid-cols-2 gap-4">
                <div className="rounded-2xl border border-white/6 bg-black/20 overflow-hidden">
                  <div className="px-4 py-3 border-b border-white/6 bg-white/4">
                    <p className="text-sm font-medium text-text-primary">Saida do Lab</p>
                    <p className="text-xs text-text-secondary mt-1">
                      {referencePreview?.output_kind === "source_fallback"
                        ? "Fallback EN ate existir output persistido do laboratorio."
                        : "Pagina processada pelo Lab."}
                    </p>
                  </div>
                  <div className="h-[360px] flex items-center justify-center p-4 bg-[radial-gradient(circle_at_top,_rgba(120,119,198,0.18),_rgba(7,10,21,0.88)_55%)]">
                    {outputImage.imageUrl ? (
                      <img src={outputImage.imageUrl} alt="Saida do Lab" className="max-h-full max-w-full object-contain rounded-xl shadow-2xl" />
                    ) : (
                      <p className="text-xs text-text-secondary px-6 text-center">
                        {outputImage.error ?? "Nenhuma pagina processada disponivel ainda."}
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/6 bg-black/20 overflow-hidden">
                  <div className="px-4 py-3 border-b border-white/6 bg-white/4">
                    <p className="text-sm font-medium text-text-primary">Scan PT-BR de referencia</p>
                    <p className="text-xs text-text-secondary mt-1">
                      {selectedPair ? `${selectedPair.reference_group} - capitulo ${selectedPair.chapter_number}` : "Selecione um capitulo valido."}
                    </p>
                  </div>
                  <div className="h-[360px] flex items-center justify-center p-4 bg-[radial-gradient(circle_at_top,_rgba(120,119,198,0.18),_rgba(7,10,21,0.88)_55%)]">
                    {referenceImage.imageUrl ? (
                      <img src={referenceImage.imageUrl} alt="Referencia PT-BR" className="max-h-full max-w-full object-contain rounded-xl shadow-2xl" />
                    ) : (
                      <p className="text-xs text-text-secondary px-6 text-center">
                        {referenceImage.error ?? "Referencia indisponivel para o capitulo selecionado."}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeSection === "run" && (
          <section className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Corpus do run</p>
              <h2 className="text-lg font-semibold text-text-primary mt-1">
                {routeTargetId ? `Run ${routeTargetId}` : "Run atual do laboratorio"}
              </h2>
            </div>
            <div className="overflow-hidden rounded-2xl border border-white/6">
              <table className="w-full text-sm">
                <thead className="bg-white/4 text-text-secondary">
                  <tr>
                    <th className="text-left px-4 py-3">Capitulo</th>
                    <th className="text-left px-4 py-3">Origem EN</th>
                    <th className="text-left px-4 py-3">Referencia PT-BR</th>
                    <th className="text-left px-4 py-3">Paginas</th>
                    <th className="text-left px-4 py-3">Grupo</th>
                  </tr>
                </thead>
                <tbody>
                  {(snapshot?.chapter_pairs ?? []).map((pair) => (
                    <tr key={pair.chapter_number} className="border-t border-white/6">
                      <td className="px-4 py-3 text-text-primary">Cap. {pair.chapter_number}</td>
                      <td className="px-4 py-3 text-text-secondary truncate max-w-[280px]">{pair.source_path}</td>
                      <td className="px-4 py-3 text-text-secondary truncate max-w-[280px]">{pair.reference_path}</td>
                      <td className="px-4 py-3 text-text-secondary">{pair.source_pages} / {pair.reference_pages}</td>
                      <td className="px-4 py-3 text-text-secondary">{pair.reference_group}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {(activeSection === "reviews" || activeSection === "decisions") && (
          <section className="space-y-4">
            <div className="grid xl:grid-cols-2 gap-4">
              {((activeSection === "decisions" ? snapshot?.proposals ?? [] : selectedProposal ? [selectedProposal] : [])).map((proposal) => {
                const proposalReviews = snapshot?.reviews.filter((review) => review.proposal_id === proposal.proposal_id) ?? [];
                const canApprove = proposal.proposal_status === "benchmark_passed" && proposal.integration_verdict === "approve";
                return (
                  <div key={proposal.proposal_id} className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <button
                          onClick={() => {
                            setHighlightedProposalId(proposal.proposal_id);
                            navigate(`/lab/reviews/${proposal.proposal_id}`);
                          }}
                          className="text-left"
                        >
                          <p className="text-lg font-semibold text-text-primary hover:text-accent-purple transition-smooth">{proposal.title}</p>
                        </button>
                        <p className="text-sm text-text-secondary mt-2">{proposal.summary}</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(proposal.proposal_status)}`}>
                        {labelForProposalStatus(proposal.proposal_status)}
                      </span>
                    </div>

                    <div className="grid md:grid-cols-2 gap-3 text-xs">
                      <div className="rounded-2xl border border-white/6 bg-black/20 p-3">
                        <p className="text-text-secondary">Autor e risco</p>
                        <p className="text-text-primary mt-1">{proposal.author}</p>
                        <p className="text-text-secondary mt-2">Risco: {proposal.risk}</p>
                      </div>
                      <div className="rounded-2xl border border-white/6 bg-black/20 p-3">
                        <p className="text-text-secondary">Integracao e PR</p>
                        <p className="text-text-primary mt-1">{labelForVerdict(proposal.integration_verdict || "pending")}</p>
                        <p className="text-text-secondary mt-2">{proposal.pr_status || (proposal.git_available ? "aguardando" : "bloqueado sem git")}</p>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {proposal.touched_domains.map((domain) => (
                        <span key={`${proposal.proposal_id}-${domain}`} className="px-2 py-1 rounded-full bg-white/6 text-[11px] text-text-secondary">
                          {domain}
                        </span>
                      ))}
                    </div>

                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Revisores</p>
                      <div className="flex flex-wrap gap-2">
                        {proposal.required_reviewers.map((reviewerId) => {
                          const review = proposalReviews.find((candidate) => candidate.reviewer_id === reviewerId);
                          return (
                            <span key={`${proposal.proposal_id}-${reviewerId}`} className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(review?.verdict || "pending")}`}>
                              {compactReviewerLabel(reviewerId)}: {labelForVerdict(review?.verdict || "pending")}
                            </span>
                          );
                        })}
                      </div>
                    </div>

                    {proposal.review_findings.length > 0 && (
                      <div className="space-y-2">
                        {proposal.review_findings.slice(0, 4).map((finding, index) => (
                          <div key={`${proposal.proposal_id}-${index}`} className="rounded-2xl border border-white/6 bg-black/20 p-3">
                            <div className="flex items-center justify-between gap-3">
                              <p className="text-sm text-text-primary">{finding.title}</p>
                              <span className={`px-2 py-0.5 rounded-full text-[10px] ${toneForStatus(finding.severity)}`}>
                                {finding.severity || "info"}
                              </span>
                            </div>
                            <p className="text-xs text-text-secondary mt-2">{finding.body}</p>
                            {finding.file_path && (
                              <p className="text-[11px] text-accent-cyan mt-2">{finding.file_path}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Planner metadata */}
                    {(proposal.motivation || proposal.change_kind) && (
                      <div className="rounded-2xl border border-white/6 bg-black/20 p-3 space-y-1.5 text-xs">
                        {proposal.motivation && (
                          <p className="text-text-secondary leading-relaxed">{proposal.motivation}</p>
                        )}
                        <div className="flex flex-wrap gap-2 pt-1">
                          {proposal.change_kind && (
                            <span className="px-2 py-0.5 rounded-full bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-[10px]">
                              {proposal.change_kind}
                            </span>
                          )}
                          {proposal.issue_type && (
                            <span className="px-2 py-0.5 rounded-full bg-white/6 border border-white/10 text-text-secondary text-[10px]">
                              {proposal.issue_type}
                            </span>
                          )}
                          {proposal.target_file && (
                            <span className="px-2 py-0.5 rounded-full bg-white/6 border border-white/10 text-accent-cyan text-[10px] font-mono">
                              {proposal.target_file}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Patch preview se ja gerado */}
                    {proposal.patch_proposal && (
                      <button
                        onClick={() => setPatchModal({ proposalId: proposal.proposal_id, patch: proposal.patch_proposal! })}
                        className="w-full text-left rounded-2xl border border-accent-cyan/20 bg-accent-cyan/5 p-3 text-xs hover:bg-accent-cyan/10 transition-smooth"
                      >
                        <p className="text-accent-cyan font-medium">
                          Patch gerado ({proposal.patch_proposal.author}) · confianca {Math.round(proposal.patch_proposal.confidence * 100)}%
                        </p>
                        <p className="text-text-secondary mt-1 truncate">
                          {proposal.patch_proposal.rationale || proposal.patch_proposal.error || "Ver diff…"}
                        </p>
                      </button>
                    )}

                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => handleApproveProposal(proposal.proposal_id)}
                        disabled={!canApprove || busyAction !== null}
                        className="px-3 py-2 rounded-xl bg-status-success/15 border border-status-success/25 text-status-success text-xs font-medium hover:bg-status-success/20 transition-smooth disabled:opacity-40"
                      >
                        Aprovar proposta
                      </button>
                      <button
                        onClick={() => handleRejectProposal(proposal.proposal_id)}
                        disabled={busyAction !== null}
                        className="px-3 py-2 rounded-xl bg-status-error/12 border border-status-error/20 text-status-error text-xs font-medium hover:bg-status-error/18 transition-smooth disabled:opacity-40"
                      >
                        Rejeitar proposta
                      </button>
                      <button
                        onClick={() => handleGeneratePatch(proposal.proposal_id)}
                        disabled={busyAction !== null}
                        className="px-3 py-2 rounded-xl bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-xs font-medium hover:bg-accent-purple/15 transition-smooth disabled:opacity-40"
                        title={`Estrategia: ${patchCoderStrategy}`}
                      >
                        {busyAction === `patch:${proposal.proposal_id}` ? "Gerando…" : "Gerar patch"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {activeSection === "reviews" && selectedReviews.length > 0 && (
              <div className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Pareceres</p>
                  <h2 className="text-lg font-semibold text-text-primary mt-1">Senior reviewers e integrador</h2>
                </div>
                <div className="grid xl:grid-cols-2 gap-4">
                  {selectedReviews.map((review) => (
                    <div key={`${review.proposal_id}-${review.reviewer_id}`} className="rounded-2xl border border-white/6 bg-black/20 p-4 space-y-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-medium text-text-primary">{review.reviewer_label}</p>
                        <span className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(review.verdict)}`}>
                          {labelForVerdict(review.verdict)}
                        </span>
                      </div>
                      <p className="text-xs text-text-secondary">Emitido em {formatTimestamp(review.reviewed_at_ms)}</p>
                      {review.findings.map((finding, index) => (
                        <div key={`${review.reviewer_id}-${index}`} className="rounded-xl border border-white/6 bg-white/4 p-3">
                          <p className="text-sm text-text-primary">{finding.title}</p>
                          <p className="text-xs text-text-secondary mt-2">{finding.body}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {activeSection === "benchmarks" && (
          <section className="grid xl:grid-cols-2 gap-4">
            {(snapshot?.benchmarks ?? []).map((benchmark) => (
              <div key={benchmark.proposal_id} className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Benchmark</p>
                    <h2 className="text-lg font-semibold text-text-primary mt-1">{benchmark.proposal_id}</h2>
                    <p className="text-xs text-text-secondary mt-1">{benchmark.summary}</p>
                  </div>
                  <span className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(benchmark.green ? "approve" : "block")}`}>
                    {benchmark.green ? "Verde" : "Falhou"}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-2xl border border-white/6 bg-black/20 p-4">
                    <p className="text-xs text-text-secondary">Score antes</p>
                    <p className="text-2xl font-semibold text-text-primary mt-2">{benchmark.score_before.toFixed(1)}</p>
                  </div>
                  <div className="rounded-2xl border border-white/6 bg-black/20 p-4">
                    <p className="text-xs text-text-secondary">Score depois</p>
                    <p className="text-2xl font-semibold text-text-primary mt-2">{benchmark.score_after.toFixed(1)}</p>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <p className="text-text-secondary">Similaridade textual: <span className="text-text-primary">{benchmark.metrics.textual_similarity.toFixed(1)}%</span></p>
                  <p className="text-text-secondary">Consistencia de termos: <span className="text-text-primary">{benchmark.metrics.term_consistency.toFixed(1)}%</span></p>
                  <p className="text-text-secondary">Ocupacao do balao: <span className="text-text-primary">{benchmark.metrics.layout_occupancy.toFixed(1)}%</span></p>
                  <p className="text-text-secondary">Legibilidade: <span className="text-text-primary">{benchmark.metrics.readability.toFixed(1)}%</span></p>
                  <p className="text-text-secondary">Limpeza visual: <span className="text-text-primary">{benchmark.metrics.visual_cleanup.toFixed(1)}%</span></p>
                  <p className="text-text-secondary">Edicoes poupadas: <span className="text-text-primary">{benchmark.metrics.manual_edits_saved.toFixed(1)}%</span></p>
                </div>
              </div>
            ))}
          </section>
        )}

        {activeSection === "history" && (
          <section className="rounded-3xl border border-white/6 bg-white/4 p-5 space-y-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-text-secondary">Historico do Lab</p>
              <h2 className="text-lg font-semibold text-text-primary mt-1">Rodadas recentes</h2>
            </div>
            <div className="space-y-3">
              {(snapshot?.history ?? []).slice().reverse().map((entry) => (
                <div key={entry.run_id} className="rounded-2xl border border-white/6 bg-black/20 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-text-primary">{entry.run_id}</p>
                      <p className="text-xs text-text-secondary mt-1">{entry.summary}</p>
                    </div>
                    <span className={`px-2.5 py-1 rounded-full border text-[11px] ${toneForStatus(entry.status)}`}>
                      {entry.status}
                    </span>
                  </div>
                  <div className="grid md:grid-cols-3 gap-3 mt-4 text-xs">
                    <div className="rounded-xl border border-white/6 bg-white/4 px-3 py-2">
                      <p className="text-text-secondary">Pares processados</p>
                      <p className="text-text-primary mt-1">{entry.processed_pairs} / {entry.total_pairs}</p>
                    </div>
                    <div className="rounded-xl border border-white/6 bg-white/4 px-3 py-2">
                      <p className="text-text-secondary">Inicio</p>
                      <p className="text-text-primary mt-1">{formatTimestamp(entry.started_at_ms)}</p>
                    </div>
                    <div className="rounded-xl border border-white/6 bg-white/4 px-3 py-2">
                      <p className="text-text-secondary">Fim</p>
                      <p className="text-text-primary mt-1">{formatTimestamp(entry.finished_at_ms)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* Patch modal — dry-run diff viewer */}
      {patchModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={() => setPatchModal(null)}
        >
          <div
            className="relative w-full max-w-3xl max-h-[90vh] rounded-3xl border border-white/10 bg-bg-surface flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-5 border-b border-white/8">
              <div>
                <p className="text-sm font-semibold text-text-primary">Patch gerado — dry-run</p>
                <p className="text-xs text-text-secondary mt-0.5">
                  {patchModal.patch.author} · modelo: {patchModal.patch.model_used || "local"} · confiança {Math.round(patchModal.patch.confidence * 100)}%
                </p>
              </div>
              <button
                onClick={() => setPatchModal(null)}
                className="p-2 rounded-xl text-text-secondary hover:text-text-primary hover:bg-white/8 transition-smooth"
              >
                <XCircle size={18} />
              </button>
            </div>

            {/* Coder strategy selector */}
            <div className="flex items-center gap-3 px-5 py-3 border-b border-white/8 text-xs">
              <span className="text-text-secondary">Coder:</span>
              {(["local", "ollama", "claude_code", "claude_sdk"] as LabCoderStrategy[]).map((s) => (
                <button
                  key={s}
                  onClick={() => setPatchCoderStrategy(s)}
                  className={`px-2.5 py-1 rounded-full border text-[11px] transition-smooth ${patchCoderStrategy === s ? "border-accent-purple/50 bg-accent-purple/15 text-accent-purple" : "border-white/10 bg-white/4 text-text-secondary"}`}
                >
                  {s === "local" ? "Local (sem LLM)" : s === "ollama" ? "Ollama" : s === "claude_code" ? "Claude Code" : "Claude SDK"}
                </button>
              ))}
              <button
                onClick={() => handleGeneratePatch(patchModal.proposalId)}
                disabled={busyAction !== null}
                className="ml-auto px-3 py-1.5 rounded-xl bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-[11px] font-medium hover:bg-accent-purple/18 disabled:opacity-40 transition-smooth"
              >
                {busyAction === `patch:${patchModal.proposalId}` ? "Gerando…" : "Regenerar"}
              </button>
            </div>

            {/* Rationale */}
            {patchModal.patch.rationale && (
              <div className="px-5 py-3 border-b border-white/8">
                <p className="text-xs text-text-secondary leading-relaxed">{patchModal.patch.rationale}</p>
              </div>
            )}

            {/* Error */}
            {patchModal.patch.error && (
              <div className="px-5 py-3 border-b border-white/8">
                <p className="text-xs text-status-error">{patchModal.patch.error}</p>
              </div>
            )}

            {/* Diff */}
            <div className="flex-1 overflow-y-auto p-5">
              {patchModal.patch.patch_unified_diff ? (
                <pre className="text-[11px] font-mono leading-relaxed whitespace-pre-wrap break-all">
                  {patchModal.patch.patch_unified_diff.split("\n").map((line, i) => (
                    <span
                      key={i}
                      className={
                        line.startsWith("+") && !line.startsWith("+++")
                          ? "text-status-success block"
                          : line.startsWith("-") && !line.startsWith("---")
                          ? "text-status-error block"
                          : line.startsWith("@@")
                          ? "text-accent-cyan block"
                          : "text-text-secondary block"
                      }
                    >
                      {line || " "}
                    </span>
                  ))}
                </pre>
              ) : (
                <p className="text-xs text-text-secondary italic">
                  Nenhum diff gerado. Tente outro coder ou verifique os logs.
                </p>
              )}
            </div>

            {/* Footer — ações de aplicação */}
            <div className="px-5 py-4 border-t border-white/8 space-y-3">
              {/* Resultado de sucesso */}
              {applyResult?.applied && (
                <div className="rounded-xl border border-status-success/25 bg-status-success/8 p-3 text-xs space-y-1">
                  <p className="text-status-success font-medium">✓ Patch aplicado com sucesso</p>
                  {applyResult.branch_created && (
                    <p className="text-text-secondary">
                      Branch: <code className="text-accent-cyan">{applyResult.branch_created}</code>
                      {applyResult.commit_sha && (
                        <span> · commit <code className="text-accent-cyan">{applyResult.commit_sha}</code></span>
                      )}
                    </p>
                  )}
                  {applyResult.files_patched.length > 0 && (
                    <p className="text-text-secondary">
                      Arquivos: {applyResult.files_patched.join(", ")}
                    </p>
                  )}
                  {!applyResult.branch_created && (
                    <p className="text-text-secondary">
                      Patch aplicado diretamente (sem git). Revise com <code className="text-accent-cyan">git diff</code>.
                    </p>
                  )}
                </div>
              )}

              {/* Erro de aplicação */}
              {applyResult && !applyResult.applied && applyResult.error && (
                <div className="rounded-xl border border-status-error/25 bg-status-error/8 p-3 text-xs">
                  <p className="text-status-error font-medium">Falha ao aplicar patch</p>
                  <p className="text-text-secondary mt-1">{applyResult.error}</p>
                </div>
              )}

              {/* Confirmação de aplicação */}
              {applyConfirm && patchModal && (
                <div className="rounded-xl border border-accent-yellow/20 bg-accent-yellow/6 p-3 text-xs space-y-2">
                  <p className="text-accent-yellow font-medium">Confirmar aplicação do patch?</p>
                  <p className="text-text-secondary">
                    Esta ação modifica arquivos no disco. Recomendado com "Criar branch".
                  </p>
                  <div className="flex gap-2 pt-1">
                    <button
                      onClick={() => handleApplyPatch(patchModal.proposalId, patchModal.patch.patch_unified_diff, true)}
                      disabled={busyAction !== null}
                      className="px-3 py-1.5 rounded-xl bg-status-success/15 border border-status-success/25 text-status-success text-[11px] font-medium hover:bg-status-success/20 disabled:opacity-40 transition-smooth"
                    >
                      {busyAction === `apply:${patchModal.proposalId}` ? "Aplicando…" : "Aplicar + criar branch"}
                    </button>
                    <button
                      onClick={() => handleApplyPatch(patchModal.proposalId, patchModal.patch.patch_unified_diff, false)}
                      disabled={busyAction !== null}
                      className="px-3 py-1.5 rounded-xl bg-white/6 border border-white/10 text-text-secondary text-[11px] hover:bg-white/10 disabled:opacity-40 transition-smooth"
                    >
                      Aplicar direto (sem branch)
                    </button>
                    <button
                      onClick={() => setApplyConfirm(false)}
                      className="px-3 py-1.5 rounded-xl text-text-secondary text-[11px] hover:text-text-primary transition-smooth"
                    >
                      Cancelar
                    </button>
                  </div>
                </div>
              )}

              {/* Botão principal + nota */}
              {!applyConfirm && !applyResult?.applied && (
                <div className="flex items-center justify-between gap-3">
                  <p className="text-[11px] text-text-secondary">
                    Dry-run — aplique manualmente via{" "}
                    <code className="text-accent-cyan">git apply</code> ou use o botão abaixo.
                  </p>
                  {patchModal?.patch.patch_unified_diff && !patchModal.patch.error && (
                    <button
                      onClick={() => setApplyConfirm(true)}
                      disabled={busyAction !== null}
                      className="shrink-0 px-3 py-1.5 rounded-xl bg-status-success/12 border border-status-success/20 text-status-success text-[11px] font-medium hover:bg-status-success/18 disabled:opacity-40 transition-smooth"
                    >
                      Aplicar patch ↗
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
