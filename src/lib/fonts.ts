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
  astronautCity: {
    key: "astronautCity",
    cssFamily: "Astronaut City",
    source: "bundle",
    files: { regular: "/fonts/Astronaut City.ttf" },
  },
  ccWildWords: {
    key: "ccWildWords",
    cssFamily: "CC Wild Words",
    source: "bundle",
    files: {
      regular: "/fonts/Cc-Wild-Words-Roman-Font.ttf",
      italic: "/fonts/Wild-Words-Font/CC Wild Words Italic.ttf",
      boldItalic: "/fonts/Wild-Words-Font/CC Wild Words Bold Italic.ttf",
    },
  },
  leagueGothic: {
    key: "leagueGothic",
    cssFamily: "League Gothic",
    source: "bundle",
    files: { regular: "/fonts/LeagueGothic-Regular-VariableFont_wdth.ttf" },
  },
  ccTotallyAwesome: {
    key: "ccTotallyAwesome",
    cssFamily: "CC Totally Awesome",
    source: "bundle",
    files: { bold: "/fonts/commercial/CCTotallyAwesome W00 Bold.ttf" },
  },
  atma: {
    key: "atma",
    cssFamily: "Atma",
    source: "bundle",
    files: { regular: "/fonts/google/Atma-SemiBold.ttf" },
  },
  bangers: {
    key: "bangers",
    cssFamily: "Bangers",
    source: "bundle",
    files: { regular: "/fonts/google/Bangers-Regular.ttf" },
  },
  deliusUnicase: {
    key: "deliusUnicase",
    cssFamily: "Delius Unicase",
    source: "bundle",
    files: { bold: "/fonts/google/DeliusUnicase-Bold.ttf" },
  },
  luckiestGuy: {
    key: "luckiestGuy",
    cssFamily: "Luckiest Guy",
    source: "bundle",
    files: { regular: "/fonts/google/LuckiestGuy-Regular.ttf" },
  },
  permanentMarker: {
    key: "permanentMarker",
    cssFamily: "Permanent Marker",
    source: "bundle",
    files: { regular: "/fonts/google/PermanentMarker-Regular.ttf" },
  },
  readyForAnything: {
    key: "readyForAnything",
    cssFamily: "Ready for Anything BB",
    source: "bundle",
    files: {
      regular: "/fonts/Ready For Anything BB/ReadyforAnythingBB-Regular.ttf",
      bold: "/fonts/Ready For Anything BB/ReadyforAnythingBB-Bold.ttf",
      italic: "/fonts/Ready For Anything BB/ReadyforAnythingBB-Italic.ttf",
      boldItalic: "/fonts/Ready For Anything BB/ReadyforAnythingBB-BoldItalic.ttf",
    },
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
    const fontFaceUrl = (path: string) => `url("${encodeURI(path)}")`;
    for (const entry of Object.values(BUNDLE_FONTS)) {
      if (entry.files.regular) {
        const ff = new FontFace(entry.cssFamily, `${fontFaceUrl(entry.files.regular)}`, {
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
        const ff = new FontFace(entry.cssFamily, `${fontFaceUrl(entry.files.bold)}`, {
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
        const ff = new FontFace(entry.cssFamily, `${fontFaceUrl(entry.files.italic)}`, {
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
        const ff = new FontFace(entry.cssFamily, `${fontFaceUrl(entry.files.boldItalic)}`, {
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
  // Lookup por filename em todas as variantes locais.
  const filename = legacyName.replaceAll(String.fromCharCode(92), "/").split("/").pop()?.toLowerCase();
  for (const entry of Object.values(BUNDLE_FONTS)) {
    if (Object.values(entry.files).some((path) => path?.split("/").pop()?.toLowerCase() === filename)) {
      return entry.cssFamily;
    }
  }
  const googleFamily = resolveGoogleFontFilename(legacyName);
  if (googleFamily) return googleFamily;
  const systemMatch = /^SystemFont__(.+)__[^.]+\.(?:ttf|otf)$/i.exec(legacyName);
  if (systemMatch) return systemMatch[1].replace(/_/g, " ");
  // Heurísticas comuns
  if (/comic\s*neue/i.test(stripped)) return BUNDLE_FONTS.comicNeue.cssFamily;
  if (/newrotic/i.test(stripped)) return BUNDLE_FONTS.newrotic.cssFamily;
  if (/komikax/i.test(stripped)) return BUNDLE_FONTS.komikax.cssFamily;
  if (/cc\s*dave|gibbons/i.test(stripped)) return BUNDLE_FONTS.ccDaveGibbons.cssFamily;
  if (/cc[\s-]*wild[\s-]*words/i.test(stripped)) return BUNDLE_FONTS.ccWildWords.cssFamily;
  // Nome desconhecido = retorna como veio (browser tentará system match)
  return stripped;
}
