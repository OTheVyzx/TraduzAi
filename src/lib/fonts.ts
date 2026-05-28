import { resolveGoogleFontFilename } from "./googleFontsCatalog";

/**
 * FONT_REGISTRY centraliza o mapeamento entre identificador interno (FontKey) e
 * nome `cssFamily` que tanto o `@font-face` (em globals.css) quanto o `Konva.Text`
 * vão usar. Manter em um único lugar previne o bug histórico em que o nome
 * passado para Konva não batia com o `@font-face` e o navegador usava fallback
 * silencioso (Comic Neue não aparecia, por exemplo).
 *
 * Estrutura:
 *  - source 'bundle': arquivos servidos por Vite a partir de `public/fonts/`.
 *  - source 'project': arquivos importados pelo usuário, copiados para
 *    `data/projects/<id>/fonts/`. Carregados dinamicamente.
 *  - source 'system': fonte instalada no SO; sem path local. Resolvida pelo
 *    nome via `queryLocalFonts` ou Rust fallback.
 *  - source 'google': fonte do catalogo Google Fonts local, resolvida para um
 *    filename de cache estavel sem chamada runtime a API Google.
 */

export type FontSource = "bundle" | "project" | "system" | "google";

export interface FontFiles {
  regular?: string;
  bold?: string;
  italic?: string;
  boldItalic?: string;
}

export interface FontEntry {
  key: string;
  cssFamily: string;
  source: FontSource;
  files: FontFiles;
}

/** Fontes que vêm com o app (em `public/fonts/`). */
export const BUNDLE_FONTS: Record<string, FontEntry> = {
  comicNeue: {
    key: "comicNeue",
    cssFamily: "Comic Neue",
    source: "bundle",
    files: {
      regular: "/fonts/ComicNeue-Regular.ttf",
      bold: "/fonts/ComicNeue-Bold.ttf",
    },
  },
  newrotic: {
    key: "newrotic",
    cssFamily: "Newrotic",
    source: "bundle",
    files: { regular: "/fonts/Newrotic.ttf" },
  },
  komikax: {
    key: "komikax",
    cssFamily: "KOMIKAX",
    source: "bundle",
    files: { regular: "/fonts/KOMIKAX_.ttf" },
  },
  ccDaveGibbons: {
    key: "ccDaveGibbons",
    cssFamily: "CC Dave Gibbons",
    source: "bundle",
    files: { regular: "/fonts/CCDaveGibbonsLower W00 Regular.ttf" },
  },
};

/**
 * Cache de fontes registradas em runtime (project + system imports).
 * Evita re-registrar a mesma family em chamadas repetidas.
 */
const registeredFamilies = new Set<string>(Object.values(BUNDLE_FONTS).map((f) => f.cssFamily));
let bundleFontsPreloadPromise: Promise<void> | null = null;

/**
 * Carrega todas as bundle fonts via FontFace API e aguarda `document.fonts.ready`.
 *
 * Sem esta etapa, o primeiro Konva.Text pode renderizar com fonte de fallback
 * antes do navegador terminar o download do TTF — bug visual sutil que ficou
 * latente até a Fase 2 do refactor.
 */
export async function preloadEditorFonts(): Promise<void> {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  if (bundleFontsPreloadPromise) return bundleFontsPreloadPromise;

  bundleFontsPreloadPromise = (async () => {
    const loaders: Promise<FontFace>[] = [];
    for (const entry of Object.values(BUNDLE_FONTS)) {
      if (entry.files.regular) {
        const ff = new FontFace(entry.cssFamily, `url(${entry.files.regular})`, {
          weight: "400",
          style: "normal",
          display: "block",
        });
        loaders.push(ff.load().then((loaded) => {
          document.fonts.add(loaded);
          return loaded;
        }));
      }
      if (entry.files.bold) {
        const ff = new FontFace(entry.cssFamily, `url(${entry.files.bold})`, {
          weight: "700",
          style: "normal",
          display: "block",
        });
        loaders.push(ff.load().then((loaded) => {
          document.fonts.add(loaded);
          return loaded;
        }));
      }
      if (entry.files.italic) {
        const ff = new FontFace(entry.cssFamily, `url(${entry.files.italic})`, {
          weight: "400",
          style: "italic",
          display: "block",
        });
        loaders.push(ff.load().then((loaded) => {
          document.fonts.add(loaded);
          return loaded;
        }));
      }
      if (entry.files.boldItalic) {
        const ff = new FontFace(entry.cssFamily, `url(${entry.files.boldItalic})`, {
          weight: "700",
          style: "italic",
          display: "block",
        });
        loaders.push(ff.load().then((loaded) => {
          document.fonts.add(loaded);
          return loaded;
        }));
      }
    }

    await Promise.allSettled(loaders);
    await document.fonts.ready;
  })();

  return bundleFontsPreloadPromise;
}

export async function ensureEditorFontLoaded(
  fontFamily: string,
  fontSize: number,
  fontStyle = "normal",
): Promise<void> {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  await preloadEditorFonts();
  const cssStyle = /\bitalic\b/i.test(fontStyle) ? "italic" : "normal";
  const cssWeight = /\bbold\b/i.test(fontStyle) ? "700" : "400";
  try {
    await document.fonts.load(`${cssStyle} ${cssWeight} ${Math.max(8, fontSize)}px "${fontFamily}"`);
    await document.fonts.ready;
  } catch (err) {
    console.warn("[fonts] falha ao carregar fonte do editor:", fontFamily, fontStyle, err);
  }
}

/**
 * Registra uma fonte importada manualmente (`.ttf`/`.otf` selecionado pelo user).
 * Recebe os bytes já lidos do arquivo (via Tauri filesystem) e a `cssFamily`
 * desejada. Adiciona ao `document.fonts` e ao registro interno.
 */
export async function registerImportedFont(
  cssFamily: string,
  bytes: ArrayBuffer,
  weight: "400" | "700" = "400",
  style: "normal" | "italic" = "normal",
): Promise<void> {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  const ff = new FontFace(cssFamily, bytes, {
    weight,
    style,
    display: "block",
  });
  const loaded = await ff.load();
  document.fonts.add(loaded);
  registeredFamilies.add(cssFamily);
}

export async function registerRemoteFont(
  cssFamily: string,
  url: string,
  weight: "400" | "700" = "400",
  style: "normal" | "italic" = "normal",
): Promise<void> {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  const ff = new FontFace(cssFamily, `url(${url})`, {
    weight,
    style,
    display: "block",
  });
  const loaded = await ff.load();
  document.fonts.add(loaded);
  registeredFamilies.add(cssFamily);
}

/** Lista famílias bundle + projeto registradas. */
export function listLocalFontFamilies(): string[] {
  return Array.from(registeredFamilies).sort();
}

/**
 * Lista fontes do sistema usando Local Font Access API (Chromium ≥103) com
 * fallback para um Tauri command Rust quando a API não está disponível ou a
 * permissão é negada.
 */
export async function listSystemFontFamilies(): Promise<string[]> {
  const apiQuery = (window as unknown as {
    queryLocalFonts?: () => Promise<Array<{ family: string }>>;
  }).queryLocalFonts;
  if (typeof apiQuery === "function") {
    try {
      const fonts = await apiQuery();
      const families = new Set<string>();
      for (const f of fonts) families.add(f.family);
      return Array.from(families).sort();
    } catch {
      /* falls through ao fallback Rust */
    }
  }

  // Fallback Rust (implementado em Fase 2C do plano)
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const result = (await invoke("list_system_fonts")) as { family: string }[];
    return Array.from(new Set(result.map((r) => r.family))).sort();
  } catch (err) {
    console.warn("[fonts] list_system_fonts indisponível:", err);
    return [];
  }
}

/**
 * Resolve o nome legacy salvo no project.json (ex.: "ComicNeue-Bold.ttf",
 * "CCDaveGibbonsLower W00 Regular.ttf") para a `cssFamily` canônica do
 * registry. Mantém compat com projetos antigos sem schema novo.
 */
export function resolveLegacyFontFamily(legacyName: string): string {
  const stripped = legacyName.replace(/\.(ttf|otf)$/i, "").trim();
  // Lookup direto por filename
  for (const entry of Object.values(BUNDLE_FONTS)) {
    if (entry.files.regular && entry.files.regular.endsWith(legacyName)) return entry.cssFamily;
    if (entry.files.bold && entry.files.bold.endsWith(legacyName)) return entry.cssFamily;
  }
  const googleFamily = resolveGoogleFontFilename(legacyName);
  if (googleFamily) return googleFamily;
  // Heurísticas comuns
  if (/comic\s*neue/i.test(stripped)) return BUNDLE_FONTS.comicNeue.cssFamily;
  if (/newrotic/i.test(stripped)) return BUNDLE_FONTS.newrotic.cssFamily;
  if (/komikax/i.test(stripped)) return BUNDLE_FONTS.komikax.cssFamily;
  if (/cc\s*dave|gibbons/i.test(stripped)) return BUNDLE_FONTS.ccDaveGibbons.cssFamily;
  // Nome desconhecido = retorna como veio (browser tentará system match)
  return stripped;
}
