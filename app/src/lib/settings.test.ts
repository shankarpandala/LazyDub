import { describe, expect, it } from "vitest";
import {
  DEFAULTS, DEFAULT_CLONE_STRENGTH, SPEED_CAP_MAX, SPEED_CAP_MIN, STYLE_OPTIONS, TTS_SCRIPT_OPTIONS, clampSpeedCap, normalizeSettings,
  openMessage,
} from "./settings";

describe("defaults", () => {
  it("dub on the original timeline: a 1.2× speed-up cap and brief pauses allowed", () => {
    expect(DEFAULTS.speedCap).toBe(1.2);
    expect(DEFAULTS.allowFreeze).toBe(true);
  });
  it("the engine's voice match default, and everyday Telugu", () => {
    // The engine's session falls back to "closest" when `open` carries no cloneStrength: both sides must agree.
    expect(DEFAULT_CLONE_STRENGTH).toBe("closest");
    expect(DEFAULTS.cloneStrength).toBe(DEFAULT_CLONE_STRENGTH);
    expect(DEFAULTS.style).toBe("colloquial");
  });
  it("have no slow-down setting at all: the video is never slowed", () => {
    expect(Object.keys(DEFAULTS)).not.toContain("allowSlowdown");
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

describe("normalizeSettings", () => {
  it("gives the defaults for empty or broken storage", () => {
    expect(normalizeSettings({})).toEqual(DEFAULTS);
    expect(normalizeSettings(null)).toEqual(DEFAULTS);
    expect(normalizeSettings("nonsense")).toEqual(DEFAULTS);
    expect(normalizeSettings([1, 2])).toEqual(DEFAULTS);
  });

  it("migrates settings saved by an older build", () => {
    const old = {
      speedCap: 1.4, allowSlowdown: true, allowFreeze: false, maxPause: 2.5, style: "formal", voiceMode: "preset",
      prepareAhead: 120, lookahead: 1200, updateCheck: false, hud: true, theme: "light",
    };
    const s = normalizeSettings(old);
    expect(s).toEqual({
      speedCap: 1.25, allowFreeze: false, style: "formal", cloneStrength: "closest", ttsScript: "telugu",
      prepareAhead: 120, lookahead: 1200, updateCheck: false, hud: true, theme: "light",
    });
    for (const gone of ["allowSlowdown", "maxPause", "voiceMode"]) expect(s).not.toHaveProperty(gone);
  });

  it("keeps a saved voice match", () => {
    for (const c of ["closest", "balanced", "natural"] as const) expect(normalizeSettings({ cloneStrength: c }).cloneStrength).toBe(c);
  });

  it("replaces values the UI can't show with defaults", () => {
    const s = normalizeSettings({
      cloneStrength: "strong", style: "poetic", prepareAhead: 45, lookahead: "600", allowFreeze: "yes", theme: "blue", hud: 1,
      ttsScript: "roman",
    });
    expect(s.cloneStrength).toBe("closest");
    expect(s.ttsScript).toBe("telugu");
    expect(s.style).toBe("colloquial");
    expect(s.prepareAhead).toBe(300);
    expect(s.lookahead).toBe(600);
    expect(s.allowFreeze).toBe(true);
    expect(s.theme).toBe("dark");
    expect(s.hud).toBe(false);
  });

  it("round-trips through storage unchanged", () => {
    const s = normalizeSettings({ speedCap: 1.1, cloneStrength: "natural", allowFreeze: false });
    expect(normalizeSettings(JSON.parse(JSON.stringify(s)))).toEqual(s);
  });
});

describe("openMessage", () => {
  it("sends voice match, speed-up cap and pauses with the video", () => {
    const s = { ...DEFAULTS, cloneStrength: "closest" as const, speedCap: 1.1, allowFreeze: false, style: "formal" as const, lookahead: 300 };
    expect(openMessage("https://youtu.be/abc", s)).toEqual({
      type: "open", url: "https://youtu.be/abc", style: "formal", lookahead: 300,
      cloneStrength: "closest", speedCap: 1.1, allowFreeze: false, ttsScript: "telugu",
    });
  });

  it("sends the defaults when nothing was changed", () => {
    const m = openMessage("abc", DEFAULTS);
    expect(m.cloneStrength).toBe("closest");
    expect(m.speedCap).toBe(1.2);
    expect(m.allowFreeze).toBe(true);
    expect(m.ttsScript).toBe("telugu");
  });

  it("never sends a speed-up outside the engine's range, nor a slow-down flag", () => {
    const m = openMessage("abc", { ...DEFAULTS, speedCap: 1.4 });
    expect(m.speedCap).toBe(1.25);
    expect(m).not.toHaveProperty("allowSlowdown");
  });
});

describe("translation style options", () => {
  it("offer the engine's two styles, everyday first", () => {
    expect(STYLE_OPTIONS.map((o) => o.id)).toEqual(["colloquial", "formal"]);
    expect(STYLE_OPTIONS.map((o) => o.label)).toEqual(["Everyday spoken Telugu", "More formal Telugu"]);
  });
  it("are named by register, never by an amount of English", () => {
    for (const o of STYLE_OPTIONS) {
      expect(o.label).not.toMatch(/english/i);
      expect(o.hint).not.toMatch(/(fewer|more|less) English|English words|stay as they are/i);
    }
  });
  it("keep a saved style", () => {
    for (const o of STYLE_OPTIONS) expect(normalizeSettings({ style: o.id }).style).toBe(o.id);
  });
});

describe("TTS script (decision D6)", () => {
  it("defaults to Telugu script, what the voice was trained on", () => {
    expect(DEFAULTS.ttsScript).toBe("telugu");
    expect(TTS_SCRIPT_OPTIONS.map((o) => o.id)).toEqual(["telugu", "latin"]);
  });
  it("keeps a saved choice and sends it when a video opens", () => {
    const s = normalizeSettings({ ttsScript: "latin" });
    expect(s.ttsScript).toBe("latin");
    expect(openMessage("abc", s).ttsScript).toBe("latin");
  });
  it("names the Latin option as the A/B test it is", () => {
    expect(TTS_SCRIPT_OPTIONS.find((o) => o.id === "latin")!.label).toMatch(/A\/B/);
  });
});
