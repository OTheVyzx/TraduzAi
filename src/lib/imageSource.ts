export interface LoadedImageSource {
  src: string;
  revoke?: () => void;
}

export interface PreloadedImageSource extends LoadedImageSource {
  width: number;
  height: number;
}

const MAX_PRELOADED_IMAGE_SOURCES = 24;
const preloadedImageSources = new Map<string, Promise<PreloadedImageSource>>();
const preloadedImageOrder: string[] = [];

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

export function localPathFromAssetSource(path: string) {
  try {
    const url = new URL(path);
    const isAssetLocalhost =
      (url.protocol === "http:" || url.protocol === "https:") && url.hostname === "asset.localhost";
    const isAssetProtocol = url.protocol === "asset:";
    if (!isAssetLocalhost && !isAssetProtocol) return null;

    let decodedPath = decodeURIComponent(url.pathname);
    if (/^\/[A-Za-z]:[\\/]/.test(decodedPath)) {
      decodedPath = decodedPath.slice(1);
    }
    return normalizeImagePath(decodedPath);
  } catch {
    return null;
  }
}

function isViteDevLocation(location?: BrowserLocationLike | null) {
  if (!location) return false;
  if (location.protocol !== "http:" && location.protocol !== "https:") return false;
  return (
    location.hostname === "localhost" ||
    location.hostname === "127.0.0.1" ||
    location.hostname === "::1"
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

async function loadLocalImageAsBlob(path: string, type: string): Promise<LoadedImageSource> {
  const { readFile } = await import("@tauri-apps/plugin-fs");
  const bytes = await readFile(path);
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  return {
    src: url,
    revoke: () => URL.revokeObjectURL(url),
  };
}

async function loadTauriAssetSource(path: string, version: number): Promise<LoadedImageSource | null> {
  try {
    const { convertFileSrc } = await import("@tauri-apps/api/core");
    return { src: cacheBustImageSource(convertFileSrc(path), version) };
  } catch {
    return null;
  }
}

function imageSourceCacheKey(path: string, type: string, version: number) {
  return `${normalizeImagePath(path)}\n${type}\n${version}`;
}

function forgetPreloadedImageSource(key: string) {
  preloadedImageSources.delete(key);
  const index = preloadedImageOrder.indexOf(key);
  if (index >= 0) preloadedImageOrder.splice(index, 1);
}

function rememberPreloadedImageSource(key: string) {
  const existingIndex = preloadedImageOrder.indexOf(key);
  if (existingIndex >= 0) preloadedImageOrder.splice(existingIndex, 1);
  preloadedImageOrder.push(key);

  while (preloadedImageOrder.length > MAX_PRELOADED_IMAGE_SOURCES) {
    const staleKey = preloadedImageOrder.shift();
    if (!staleKey || staleKey === key) continue;
    const staleSource = preloadedImageSources.get(staleKey);
    preloadedImageSources.delete(staleKey);
    staleSource?.then((loaded) => loaded.revoke?.()).catch(() => {});
  }
}

function decodeLoadedImage(src: string): Promise<{ width: number; height: number }> {
  if (typeof Image === "undefined") {
    return Promise.resolve({ width: 0, height: 0 });
  }

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("imagem indisponivel"));
    image.src = src;
  });
}

async function loadImageSourceUncached(path: string, type = "image/jpeg", version = 0): Promise<LoadedImageSource> {
  const localAssetPath = localPathFromAssetSource(path);
  const viteDevLocation = typeof window !== "undefined" ? window.location : null;

  if (!localAssetPath && isDirectImageSource(path)) {
    return { src: cacheBustImageSource(path, version) };
  }

  const normalized = normalizeImagePath(localAssetPath ?? path);

  if (localAssetPath) {
    try {
      return await loadLocalImageAsBlob(normalized, type);
    } catch {
      // Some contexts do not grant FS access; keep protocol fallbacks below.
    }

    const tauriAssetSource = await loadTauriAssetSource(normalized, version);
    if (tauriAssetSource) {
      return tauriAssetSource;
    }

    const viteDevSource = getViteDevImageSource(normalized, viteDevLocation, version);
    if (viteDevSource) {
      return { src: viteDevSource };
    }

    try {
      return await loadLocalImageAsBlob(normalized, type);
    } catch {
      // Ultimo fallback: usa a URL asset original se o WebView tiver esse host ativo.
    }

    if (isDirectImageSource(path)) {
      return { src: cacheBustImageSource(path, version) };
    }
  }

  if (isWindowsAbsolutePath(normalized)) {
    try {
      return await loadLocalImageAsBlob(normalized, type);
    } catch {
      // Prefer blob URLs in Tauri, but keep asset/Vite fallbacks for dev/browser contexts.
    }

    const tauriAssetSource = await loadTauriAssetSource(normalized, version);
    if (tauriAssetSource) {
      return tauriAssetSource;
    }

    const viteDevSource = getViteDevImageSource(normalized, viteDevLocation, version);
    if (viteDevSource) {
      return { src: viteDevSource };
    }

    try {
      return await loadLocalImageAsBlob(normalized, type);
    } catch {
      // Fallback final; algumas permissoes de FS podem bloquear readFile.
    }
  }

  const tauriAssetSource = await loadTauriAssetSource(normalized, version);
  if (tauriAssetSource) {
    return tauriAssetSource;
  }

  try {
    return await loadLocalImageAsBlob(normalized, type);
  } catch (readError) {
    throw readError;
  }
}

export async function loadImageSource(path: string, type = "image/jpeg", version = 0): Promise<LoadedImageSource> {
  const cacheKey = imageSourceCacheKey(path, type, version);
  const preloaded = preloadedImageSources.get(cacheKey);
  if (preloaded) {
    const loaded = await preloaded;
    return { src: loaded.src };
  }

  return loadImageSourceUncached(path, type, version);
}

export async function preloadImageSource(path: string, type = "image/jpeg", version = 0) {
  const cacheKey = imageSourceCacheKey(path, type, version);
  const existing = preloadedImageSources.get(cacheKey);
  if (existing) return existing;

  const pending = loadImageSourceUncached(path, type, version)
    .then(async (loaded) => {
      try {
        const decoded = await decodeLoadedImage(loaded.src);
        return { ...loaded, ...decoded };
      } catch (error) {
        loaded.revoke?.();
        throw error;
      }
    })
    .catch((error) => {
      forgetPreloadedImageSource(cacheKey);
      throw error;
    });

  preloadedImageSources.set(cacheKey, pending);
  rememberPreloadedImageSource(cacheKey);
  return pending;
}
