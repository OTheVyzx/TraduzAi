import type {
  PerformanceTier,
  PipelineTimeEstimate,
  ProjectQuality,
  SystemProfile,
} from "./stores/appStore";

export function formatDuration(seconds: number): string {
  const safeSeconds = Math.max(0, Math.round(seconds));
  if (safeSeconds < 60) return `${safeSeconds}s`;

  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainingSeconds = safeSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  if (remainingSeconds === 0) {
    return `${minutes}m`;
  }

  return `${minutes}m ${remainingSeconds}s`;
}

export function formatEtaClock(secondsFromNow: number): string {
  const target = new Date(Date.now() + Math.max(0, secondsFromNow) * 1000);
  return new Intl.DateTimeFormat("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(target);
}

export function formatQualityLabel(quality: ProjectQuality): string {
  if (quality === "rapida") return "Rapida";
  if (quality === "alta") return "Alta";
  return "Normal";
}

export function formatTierLabel(tier: PerformanceTier): string {
  if (tier === "workstation") return "Workstation";
  if (tier === "fast") return "Rapido";
  if (tier === "balanced") return "Balanceado";
  return "CPU";
}

export function buildHardwareSummary(profile: SystemProfile | null): string {
  if (!profile) {
    return "Detectando CPU, RAM e aceleracao local...";
  }

  const accelerator = profile.gpu_available
    ? profile.gpu_vram_gb
      ? `${profile.gpu_name} (${profile.gpu_vram_gb.toFixed(1)} GB VRAM)`
      : profile.gpu_name
    : "CPU sem CUDA";

  return `${accelerator} | ${profile.cpu_threads} threads | ${profile.ram_gb} GB RAM`;
}

export function buildPipelineTimeEstimate(
  profile: SystemProfile | null,
  totalPages: number,
  quality: ProjectQuality
): PipelineTimeEstimate | null {
  if (!profile || totalPages <= 0) {
    return null;
  }

  const secondsPerPage = profile.seconds_per_page[quality];
  const totalSeconds = profile.startup_seconds + secondsPerPage * totalPages;

  return {
    total_pages: totalPages,
    quality,
    total_seconds: Math.round(totalSeconds),
    seconds_per_page: secondsPerPage,
    startup_seconds: profile.startup_seconds,
    performance_tier: profile.performance_tier,
    hardware_summary: buildHardwareSummary(profile),
  };
}

export function blendRemainingSeconds(args: {
  initialTotalSeconds: number;
  elapsedSeconds: number;
  progressPercent: number;
  liveEtaSeconds: number;
}): number {
  const progress = clamp(args.progressPercent / 100, 0, 0.995);
  const elapsed = Math.max(0, args.elapsedSeconds);
  const initialRemaining = Math.max(0, args.initialTotalSeconds - elapsed);
  const liveEta = Math.max(0, args.liveEtaSeconds);

  if (progress <= 0.01) {
    return liveEta > 0 ? weightedAverage([[initialRemaining, 0.75], [liveEta, 0.25]]) : initialRemaining;
  }

  const observedTotal = elapsed / Math.max(progress, 0.05);
  const observedRemaining = Math.max(0, observedTotal - elapsed);

  if (liveEta > 0) {
    const liveWeight = progress >= 0.5 ? 0.6 : progress >= 0.2 ? 0.5 : 0.35;
    const observedWeight = progress >= 0.2 ? 0.3 : 0.2;
    const initialWeight = Math.max(0, 1 - liveWeight - observedWeight);
    return weightedAverage([
      [liveEta, liveWeight],
      [observedRemaining, observedWeight],
      [initialRemaining, initialWeight],
    ]);
  }

  return weightedAverage([
    [observedRemaining, 0.65],
    [initialRemaining, 0.35],
  ]);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function weightedAverage(values: Array<[number, number]>): number {
  const totalWeight = values.reduce((sum, [, weight]) => sum + weight, 0);
  if (totalWeight <= 0) return 0;

  const total = values.reduce((sum, [value, weight]) => sum + value * weight, 0);
  return total / totalWeight;
}
