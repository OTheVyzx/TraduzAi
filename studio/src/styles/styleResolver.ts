import {
  colorToRgba,
  fontStyleForResolvedTextStyle,
  resolveEditorTextStyle,
  type ResolvedEditorTextStyle,
} from "../../../src/lib/editorTextStyleResolver";

export function resolveStudioTextStyle(style: unknown, legacyStyle?: unknown): ResolvedEditorTextStyle {
  return resolveEditorTextStyle(style, legacyStyle);
}

export { colorToRgba, fontStyleForResolvedTextStyle };
