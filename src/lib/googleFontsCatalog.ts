export type GoogleFontVariant = "regular" | "700";

export interface GoogleFontCatalogEntry {
  key: string;
  label: string;
  cssFamily: string;
  source: "google";
  variants: GoogleFontVariant[];
  filename: string;
  files: Partial<Record<GoogleFontVariant, string>>;
  downloadUrls: Partial<Record<GoogleFontVariant, string>>;
}

function googleFontFilename(slug: string, variant: GoogleFontVariant): string {
  return `GoogleFont__${slug}__${variant}.ttf`;
}

function googleFontEntry(
  key: string,
  label: string,
  slug: string,
  variants: GoogleFontVariant[],
  downloadUrls: Partial<Record<GoogleFontVariant, string>>,
): GoogleFontCatalogEntry {
  const files = Object.fromEntries(variants.map((variant) => [variant, googleFontFilename(slug, variant)]));
  return {
    key,
    label,
    cssFamily: label,
    source: "google",
    variants,
    filename: googleFontFilename(slug, variants[0] ?? "regular"),
    files,
    downloadUrls,
  };
}

export const GOOGLE_FONTS_CATALOG: readonly GoogleFontCatalogEntry[] = [
  googleFontEntry("googleBangers", "Bangers", "Bangers", ["regular"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
  }),
  googleFontEntry("googleComicNeue", "Comic Neue", "Comic_Neue", ["regular", "700"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Regular.ttf",
    "700": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
  }),
  googleFontEntry("googlePatrickHand", "Patrick Hand", "Patrick_Hand", ["regular"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
  }),
  googleFontEntry("googleMPlusRounded1c", "M PLUS Rounded 1c", "M_PLUS_Rounded_1c", ["regular", "700"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/mplusrounded1c/MPLUSRounded1c-Regular.ttf",
    "700": "https://raw.githubusercontent.com/google/fonts/main/ofl/mplusrounded1c/MPLUSRounded1c-Bold.ttf",
  }),
  googleFontEntry("googleNotoSans", "Noto Sans", "Noto_Sans", ["regular", "700"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "700": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
  }),
  googleFontEntry("googleNotoSerif", "Noto Serif", "Noto_Serif", ["regular", "700"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf",
    "700": "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf",
  }),
  googleFontEntry("googleRobotoCondensed", "Roboto Condensed", "Roboto_Condensed", ["regular", "700"], {
    regular: "https://raw.githubusercontent.com/google/fonts/main/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf",
    "700": "https://raw.githubusercontent.com/google/fonts/main/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf",
  }),
];

export function resolveGoogleFontFilename(filename: string): string | null {
  for (const entry of GOOGLE_FONTS_CATALOG) {
    if (entry.filename === filename) return entry.cssFamily;
    for (const variant of entry.variants) {
      if (entry.files[variant] === filename) return entry.cssFamily;
    }
  }
  const dynamicMatch = /^GoogleFont__(.+)__[^.]+\.(?:ttf|otf)$/i.exec(filename);
  if (dynamicMatch) return dynamicMatch[1].replace(/_/g, " ");
  return null;
}

export function findGoogleFontByFilename(filename: string): {
  entry: GoogleFontCatalogEntry;
  variant: GoogleFontVariant;
  filename: string;
  downloadUrl: string;
} | null {
  for (const entry of GOOGLE_FONTS_CATALOG) {
    for (const variant of entry.variants) {
      const variantFilename = entry.files[variant];
      const downloadUrl = entry.downloadUrls[variant];
      if (variantFilename === filename && downloadUrl) {
        return { entry, variant, filename: variantFilename, downloadUrl };
      }
    }
  }
  return null;
}
