import { describe, expect, it } from "vitest";
import { AppState, asStage } from "./state.svelte";
import { DEFAULTS } from "./settings";
import type { ClientMessage, Speaker } from "./types";

const speaker = (id: string, over: Partial<Speaker> = {}): Speaker => ({
  id, label: `Speaker ${id.slice(1)}`, voice: "cloned", status: "cloned", referenceSeconds: 10, talkSeconds: 60, usePreset: false, ...over,
});

/** A fake engine connection that records what the UI sends. */
function recorder() {
  const sent: ClientMessage[] = [];
  return { sent, send: (m: ClientMessage) => void sent.push(m) };
}

describe("opening a video", () => {
  it("sends the settings the engine dubs it with", () => {
    const app = new AppState();
    app.settings = { ...DEFAULTS, speedCap: 1.1, allowFreeze: false, cloneStrength: "natural", style: "formal", lookahead: 1200 };
    const { sent, send } = recorder();
    app.openVideo("https://youtu.be/abc", send);
    expect(sent).toEqual([{
      type: "open", url: "https://youtu.be/abc", style: "formal", lookahead: 1200,
      cloneStrength: "natural", speedCap: 1.1, allowFreeze: false, ttsScript: "telugu",
    }]);
  });

  it("remembers the settings the video was opened with, not later changes", () => {
    const app = new AppState();
    app.settings = { ...DEFAULTS, cloneStrength: "natural", lookahead: 300 };
    app.openVideo("abc", recorder().send);
    expect(app.cloneStrength).toBe("natural");
    expect(app.lookahead).toBe(300);
    app.settings.cloneStrength = "closest";
    app.settings.lookahead = 1200;
    expect(app.cloneStrength).toBe("natural");
    expect(app.lookahead).toBe(300);
  });

  it("forgets the previous video and shows that it's looking for the new one", () => {
    const app = new AppState();
    app.speakers = [speaker("S1")];
    app.addNotice({ kind: "fast_speech", from: 10, to: 50 });
    app.lineCount = 12;
    app.error = { message: "gone", retryable: true };
    app.openVideo("abc", recorder().send);
    expect(app.speakers).toEqual([]);
    expect(app.notices).toEqual([]);
    expect(app.lineCount).toBe(0);
    expect(app.error).toBeNull();
    expect(app.stage).toBe("resolving");
    expect(app.message).toBe("Finding the video…");
  });
});

describe("switching a speaker's voice", () => {
  it("shows the switch at once and tells the engine", () => {
    const app = new AppState();
    app.speakers = [speaker("S1"), speaker("S2")];
    const { sent, send } = recorder();
    app.setPreset("S2", true, send);
    expect(app.speakers.map((s) => s.usePreset)).toEqual([false, true]);
    expect(sent).toEqual([{ type: "speaker_preset", speaker: "S2", usePreset: true }]);
    app.setPreset("S2", false, send);
    expect(app.speakers.map((s) => s.usePreset)).toEqual([false, false]);
    expect(sent.at(-1)).toEqual({ type: "speaker_preset", speaker: "S2", usePreset: false });
  });
});

describe("engine status", () => {
  it("follows the stages this UI knows", () => {
    const app = new AppState();
    app.setStatus("listening", "Listening…");
    expect(app.stage).toBe("listening");
    expect(app.message).toBe("Listening…");
  });
  it("ignores a stage it doesn't know instead of showing a broken pipeline", () => {
    const app = new AppState();
    app.setStatus("fetching", "Fetching audio…");
    app.setStatus("fast_speech", "Fast speech ahead");
    expect(app.stage).toBe("fetching");
    expect(app.message).toBe("Fetching audio…");
    expect(asStage("fast_speech")).toBeNull();
    expect(asStage("voicing")).toBe("voicing");
  });
});

describe("notices", () => {
  it("shows a notice while the playhead is in its stretch", () => {
    const app = new AppState();
    app.addNotice({ kind: "fast_speech", from: 30, to: 70 });
    app.time = 10;
    expect(app.notice).toBeNull();
    app.time = 45;
    expect(app.notice).toEqual({ kind: "fast_speech", from: 30, to: 70 });
    app.time = 75;
    expect(app.notice).toBeNull();
  });
});

describe("Claude", () => {
  const health = { installed: true, version: "2.1.281", signedIn: true, models: ["claude-sonnet-5"], model: "claude-sonnet-5",
    problem: null, message: "" };

  it("knows from hello whether the engine translates through Claude, and what is wrong with it already", () => {
    const app = new AppState();
    app.setClaude(null);
    expect(app.claude).toBeNull();
    expect(app.claudeProblem).toBeNull();
    app.setClaude({ ...health, signedIn: false, problem: "not_signed_in", message: "Run claude auth login." }, 5000);
    expect(app.claude?.version).toBe("2.1.281");
    expect(app.claudeProblem).toEqual({ kind: "not_signed_in", message: "Run claude auth login.", at: 5000 });
    app.setClaude(health);
    expect(app.claudeProblem).toBeNull();
  });

  it("shows a problem the engine reports until a call goes through again", () => {
    const app = new AppState();
    app.claudeError({ kind: "usage_limit", message: "limit", limit: "session", resetsAt: 1790300000, retryIn: 600 }, 1000);
    expect(app.claudeProblem).toMatchObject({ kind: "usage_limit", limit: "session", at: 1000 });
    app.claudeOk();
    expect(app.claudeProblem).toBeNull();
  });

  it("remembers that the privacy notice was read", () => {
    const store = new Map<string, string>();
    const saved = globalThis.localStorage;
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
    });
    try {
      const app = new AppState();
      expect(app.privacySeen).toBe(false);
      app.ackPrivacy();
      expect(app.privacySeen).toBe(true);
      expect(new AppState().privacySeen).toBe(true);
    } finally {
      Object.defineProperty(globalThis, "localStorage", { configurable: true, value: saved });
    }
  });
});
