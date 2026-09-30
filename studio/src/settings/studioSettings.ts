import type { StudioLibrary } from "../library/libraryModel";

export interface StudioSettingsDraft {
  defaultChapterView: "grid" | "list";
  thumbnailSize: number;
  trackingLanguage: string;
}

export function createStudioSettingsDraft(preferences: StudioLibrary["preferences"]): StudioSettingsDraft {
  return {
    defaultChapterView: preferences.chapterView,
    thumbnailSize: preferences.thumbnailSize,
    trackingLanguage: preferences.trackingLanguage,
  };
}

export function resetStudioSettings(): StudioSettingsDraft {
  return { defaultChapterView: "list", thumbnailSize: 176, trackingLanguage: "en" };
}

export function isStudioSettingsDirty(current: StudioSettingsDraft, saved: StudioSettingsDraft): boolean {
  return current.defaultChapterView !== saved.defaultChapterView
    || current.thumbnailSize !== saved.thumbnailSize
    || current.trackingLanguage !== saved.trackingLanguage;
}
