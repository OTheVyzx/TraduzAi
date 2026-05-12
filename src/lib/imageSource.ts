export interface LoadedImageSource {
  src: string;
  revoke?: () => void;
}

export function normalizeImagePath(path: string) {
  return path.replace(/\\/g, "/");
}

export function isDirectImageSource(path?: string | null) {
  if (!path) return false;
  return /^(data|blob|asset|file):/i.test(path) || /^https?:\/\//i.test(path) || path.startsWith("/");
}

type BrowserLocationLike = Pick<Location, "protocol" | "hostname">;

function isWindowsAbsolutePath(path: string) {
  return /^[A-Za-z]:\//.test(path);
}

function isViteDevLocation(location?: BrowserLocationLike | null) {
  if (!location) return false;
  if (location.protocol !== "http:" && location.protocol !== "https:") return false;
  return (
    location.hostname === "localhost" ||
    location.hostname === "127.0.0.1" ||
    location.hostname === "::1" ||
    location.hostname.endsWith(".localhost")
  );
}

function encodeViteFsPath(path: string) {
  return path
    .split("/")
    .map((segment, index) => (index === 0 && /^[A-Za-z]:$/.test(segment) ? segment : encodeURIComponent(segment)))
    .join("/");
}

export function cacheBustImageSource(src: string, version = 0) {
  if (!version || /^(data|blob):/i.test(src)) return src;
  const separator = src.includes("?") ? "&" : "?";
  return `${src}${separator}v=${encodeURIComponent(String(version))}`;
}

export function getViteDevImageSource(
  path: string,
  location: BrowserLocationLike | null =
    typeof window !== "undefined" ? window.location : null,
  version = 0,
) {
  const normalized = normalizeImagePath(path);
  if (!isWindowsAbsolutePath(normalized) || !isViteDevLocation(location)) return null;
  return cacheBustImageSource(`/@fs/${encodeViteFsPath(normalized)}`, version);
}

export async function loadImageSource(path: string, type = "image/jpeg", version = 0): Promise<LoadedImageSource> {
  if (isDirectImageSource(path)) {
    return { src: cacheBustImageSource(path, version) };
  }

  const normalized = normalizeImagePath(path);
  const viteDevSource = getViteDevImageSource(normalized, undefined, version);
  if (viteDevSource) {
    return { src: viteDevSource };
  }

  try {
    const { readFile } = await import("@tauri-apps/plugin-fs");
    const bytes = await readFile(normalized);
    const url = URL.createObjectURL(new Blob([bytes], { type }));
    return {
      src: url,
      revoke: () => URL.revokeObjectURL(url),
    };
  } catch (readError) {
    try {
      const { convertFileSrc } = await import("@tauri-apps/api/core");
      return { src: cacheBustImageSource(convertFileSrc(normalized), version) };
    } catch {
      throw readError;
    }
  }
}
