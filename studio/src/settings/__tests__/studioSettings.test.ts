import { describe, expect, it } from "vitest";
import { createStudioSettingsDraft, isStudioSettingsDirty, resetStudioSettings } from "../studioSettings";

describe("studioSettings", () => {
  it("derives a persisted settings draft from real Studio library preferences", () => {
    const settings = createStudioSettingsDraft({ chapterView: "list", thumbnailSize: 192, trackingLanguage: "ko" });
    expect(settings).toEqual({ defaultChapterView: "list", thumbnailSize: 192, trackingLanguage: "ko" });
  });

  it("keeps save disabled until a real preference changes and restores defaults explicitly", () => {
    const saved = createStudioSettingsDraft({ chapterView: "list", thumbnailSize: 176, trackingLanguage: "en" });
    const edited = { ...saved, thumbnailSize: 208 };
    expect(isStudioSettingsDirty(edited, saved)).toBe(true);
    expect(resetStudioSettings()).toEqual({ defaultChapterView: "list", thumbnailSize: 176, trackingLanguage: "en" });
  });
});
