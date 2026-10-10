import { afterEach, describe, expect, it } from "vitest";
import { render } from "svelte/server";
import { app } from "../state.svelte";
import SpeakersFound from "./SpeakersFound.svelte";
import type { Render, SpeakersFound as Found } from "../types";

const found: Found = {
  videoId: "original001", mode: "auto", fresh: false, freeFor: 0, bounds: [1, 6], merged: [],
  speakers: [{ id: "S1", label: "Speaker 1", talkSeconds: 10, share: 1, firstAt: 0, turns: 1, activity: [] }],
};
const job: Render = {
  videoId: found.videoId, status: "paused", stage: "voices", stopAt: null, finalUntil: null, eta: null,
  elapsed: 1, position: null, output: null, slept: null, coverage: null, error: null,
  stages: [{ key: "voices", label: "Voices", state: "done", done: 1, total: 1, unit: "voices", seconds: 1, eta: null }],
};
const speakerCard = (presets: string[], saved: Partial<Found> = {}) =>
  render(SpeakersFound, { props: { found: { ...found, ...saved }, job, presets, send: () => {} } }).body;

afterEach(() => { app.voiceMode = "cloned"; });

describe("voice mode speaker controls", () => {
  it("offers an automatic speech sample without promising a stable cloned voice", () => {
    app.voiceMode = "native";
    const html = speakerCard(["S1"]);
    expect(html).toContain("Hear voice");
    expect(html).toContain("Natural Telugu speech");
    expect(html).toContain("without cloning source voices");
    expect(html).toContain("The automatic voice can vary between lines.");
    expect(html).not.toContain("same natural Telugu voice");
    expect(html).not.toContain('role="switch"');
    expect(html).not.toContain("Uses a clone");
  });

  it("retains the clone/stock controls for older engines", () => {
    app.voiceMode = "cloned";
    const stock = speakerCard(["S1"]);
    expect(stock).toContain('role="switch"');
    expect(stock).not.toContain("Hear voice");
    expect(speakerCard([])).toContain("Hear voice");
  });

  it("does not relabel or audition an old clone as the new native voice", () => {
    app.voiceMode = "native";
    const html = speakerCard([], { voiceMode: "cloned", voiceCompatible: false });
    expect(html).toContain("Previous voice");
    expect(html).toContain("This saved dub uses its previous voice.");
    expect(html).not.toContain("Hear voice");
    expect(html).not.toContain("Natural Telugu speech");
    expect(html).not.toContain('role="switch"');
    expect(speakerCard([], { voiceMode: "native", voiceCompatible: true })).toContain("Hear voice");
  });
});
