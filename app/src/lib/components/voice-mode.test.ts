import { afterEach, describe, expect, it } from "vitest";
import { render } from "svelte/server";
import { app } from "../state.svelte";
import SpeakersFound from "./SpeakersFound.svelte";
import TopBar from "./TopBar.svelte";
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

afterEach(() => { app.voiceMode = "cloned"; app.voiceProfileRequests = {}; app.foundBy = {}; });

describe("voice mode speaker controls", () => {
  it("describes suggestions in the top bar only after the engine exposes voice selection", () => {
    app.voiceMode = "native";
    let html = render(TopBar, { props: { onInspect: () => {} } }).body;
    expect(html).toContain("Source voices are not cloned.");
    expect(html).not.toContain("Auto suggests male or female voices");
    app.foundBy = { [found.videoId]: { ...found, speakers: [{ ...found.speakers[0]!, voiceProfile: "auto" }] } };
    html = render(TopBar, { props: { onInspect: () => {} } }).body;
    expect(html).toContain("Auto suggests male or female voices from source audio.");
    expect(html).toContain("Choose a voice when no clear match is available");
    expect(html).not.toContain("automatic voice can vary");
    app.voiceMode = "cloned";
    expect(render(TopBar, { props: { onInspect: () => {} } }).body).toContain("AI-generated from the original speakers");
  });
  it("shows editable voice choices and an acoustic suggestion without declaring speaker gender", () => {
    app.voiceMode = "native";
    const html = speakerCard([], { speakers: [{ ...found.speakers[0]!, voiceProfile: "auto",
      resolvedVoiceProfile: "female", voiceProfileSource: "acoustic" }] });
    expect(html).toContain('aria-label="Telugu voice for Speaker 1"');
    expect(html).toContain('value="auto"');
    expect(html).toContain('value="male"');
    expect(html).toContain('value="female"');
    expect(html).toContain("Auto suggestion: Female voice");
    expect(html).toContain("Hear voice");
    expect(html).toContain("Voice choice applies when you resume or dub again. Saved video stays unchanged.");
    expect(html).not.toContain("Speaker 1 is female");
  });

  it("keeps old saved dubs editable but never offers their mismatched audio as the chosen voice", () => {
    app.voiceMode = "native";
    const html = speakerCard([], { voiceMode: "cloned", voiceCompatible: false,
      speakers: [{ ...found.speakers[0]!, voiceProfile: "female", resolvedVoiceProfile: null,
        voiceProfileSource: "manual", voiceCompatible: false }] });
    expect(html).toContain('aria-label="Telugu voice for Speaker 1"');
    expect(html).toContain("Female voice selected");
    expect(html).toContain("Saved sample uses the previous voice");
    expect(html).not.toContain("Hear voice");
  });

  it("asks for a choice when Auto is unresolved and hides a stale row even if global provenance matches", () => {
    app.voiceMode = "native";
    const unresolved = { ...found.speakers[0]!, voiceProfile: "auto" as const, resolvedVoiceProfile: null,
      voiceProfileSource: "unresolved" as const };
    const html = speakerCard([], { voiceCompatible: true, speakers: [unresolved] });
    expect(html).toContain("Auto: choose a voice");
    expect(html).not.toContain("Hear voice");
    const stale = speakerCard([], { voiceCompatible: true, speakers: [{ ...unresolved,
      voiceProfile: "male", resolvedVoiceProfile: "male", voiceProfileSource: "manual", voiceCompatible: false }] });
    expect(stale).not.toContain("Hear voice");
  });

  it("does not audition the old voice while a new selection awaits its server echo", () => {
    app.voiceMode = "native";
    app.voiceProfileRequests = { [found.videoId]: { S1: "female" } };
    const html = speakerCard([], { speakers: [{ ...found.speakers[0]!, voiceProfile: "male",
      resolvedVoiceProfile: "male", voiceProfileSource: "manual" }] });
    expect(html).toContain("Saving voice choice…");
    expect(html).not.toContain("Hear voice");
  });

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
    expect(speakerCard([], { speakers: [{ ...found.speakers[0]!, voiceProfile: "auto", resolvedVoiceProfile: null,
      voiceProfileSource: "unresolved" }] })).toContain("Hear voice");
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
