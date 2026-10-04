import { describe, expect, it } from "vitest";
import {
  DEFAULT_OPTIONS, PREVIEW_AT, SPEAKER_OPTIONS, SPEED_CAP_MAX, SPEED_CAP_MIN, STYLE_OPTIONS, TTS_SCRIPT_OPTIONS, clampSpeedCap,
  jobOptions, newDubOptions,
} from "./settings";

describe("defaults", () => {
  it("auto speakers, everyday Telugu, the whole video, a 1.2× speed-up cap and Telugu script", () => {
    expect(DEFAULT_OPTIONS).toEqual({ speakers: "auto", style: "colloquial", stopAt: null, speedCap: 1.2, ttsScript: "telugu" });
    expect(SPEAKER_OPTIONS).toEqual(["auto", 1, 2, 3, 4, 5, 6]);
    expect(PREVIEW_AT).toBe(900);
  });
});

describe("clampSpeedCap", () => {
  it("keeps the range 1.0–1.25", () => {
    expect(SPEED_CAP_MIN).toBe(1);
    expect(SPEED_CAP_MAX).toBe(1.25);
    expect(clampSpeedCap(1.4)).toBe(1.25);
    expect(clampSpeedCap(0.8)).toBe(1);
    expect(clampSpeedCap(1)).toBe(1);
    expect(clampSpeedCap(1.25)).toBe(1.25);
  });
  it("snaps to the slider's 0.05 steps, exactly", () => {
    expect(clampSpeedCap(1.17)).toBe(1.15);
    expect(clampSpeedCap(1.18)).toBe(1.2);
    expect(clampSpeedCap(1.1 + 0.05)).toBe(1.15);
  });
  it("falls back to the default for anything unreadable", () => {
    expect(clampSpeedCap(Number.NaN)).toBe(1.2);
    expect(clampSpeedCap(Infinity)).toBe(1.2);
    expect(clampSpeedCap("1.1")).toBe(1.2);
    expect(clampSpeedCap(undefined)).toBe(1.2);
  });
});

describe("a job's options", () => {
  const saved = { speakers: 2, style: "formal" as const, stopAt: 900, speedCap: 1.1, ttsScript: "latin" as const, presets: ["S2"] };

  it("are what continuing it sends, its speakers included", () => {
    expect(jobOptions(saved)).toEqual({ speakers: 2, style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "latin" });
  });

  it("start a new dub from the last job's, but for the speakers: they were that video's", () => {
    expect(newDubOptions(saved)).toEqual({ speakers: "auto", style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "latin" });
    expect(newDubOptions(null)).toEqual(DEFAULT_OPTIONS);
  });

  it("replace what the UI can't offer with the defaults", () => {
    expect(jobOptions({ speakers: 9, style: "poetic" as never, stopAt: 60, speedCap: 1.4, ttsScript: "roman" as never }))
      .toEqual({ speakers: "auto", style: "colloquial", stopAt: null, speedCap: 1.25, ttsScript: "telugu" });
  });
});

describe("translation style options", () => {
  it("offer the engine's two styles, everyday first", () => {
    expect(STYLE_OPTIONS.map((o) => o.id)).toEqual(["colloquial", "formal"]);
    expect(STYLE_OPTIONS.map((o) => o.label)).toEqual(["Everyday spoken", "More formal"]);
  });
  it("are named by register, never by an amount of English", () => {
    for (const o of STYLE_OPTIONS) {
      expect(o.label).not.toMatch(/english/i);
      expect(o.hint).not.toMatch(/(fewer|more|less) English|stay as they are/i);
    }
  });
});

describe("TTS script (decision D6)", () => {
  it("defaults to Telugu script, what the voice was trained on", () => {
    expect(TTS_SCRIPT_OPTIONS.map((o) => o.id)).toEqual(["telugu", "latin"]);
  });
  it("names the Latin option as the A/B test it is", () => {
    expect(TTS_SCRIPT_OPTIONS.find((o) => o.id === "latin")!.label).toMatch(/A\/B/);
  });
});
