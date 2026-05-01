export const editorCapabilities = {
  rotacao: { enabled: false, png: "blocked" },
  line_height: { enabled: false, png: "blocked" },
  letter_spacing: { enabled: false, png: "blocked" },
  text_transform: { enabled: false, png: "blocked" },
  opacidade: { enabled: false, png: "blocked" },
} as const;

export type EditorCapabilityKey = keyof typeof editorCapabilities;
export type EditorCapability = (typeof editorCapabilities)[EditorCapabilityKey];
