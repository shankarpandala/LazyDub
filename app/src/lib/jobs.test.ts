import { describe, expect, it } from "vitest";
import {
  CLUSTERING, SLEEP_NOTE_FOR, STAGE_GROUPS, STAGE_LABELS, chip, countText, coverageShares, elapsedText, estimateLine, etaText,
  folderText, foundText, groupRows, jobsAhead, live, positionText, speakerChangeCost, untilVoicing, voiceSwitchCost,
} from "./jobs";
import { fmtReset } from "./claude";
import type { JobItem, Output, Render, StageKey, StageRow } from "./types";

/** The engine's stages in order, with their units (render.STAGES). */
const UNITS: [StageKey, string][] = [
  ["fetch", "MB"], ["speakers", "fraction"], ["transcript", "video s"], ["units", "lines"], ["voices", "speakers"],
  ["brief", "parts"], ["separate", "video s"], ["translate", "lines"], ["voice_lines", "speech s"], ["finish", "lines"],
  ["video", "MB"], ["export", "video s"],
];

const stages = (over: Partial<Record<StageKey, Partial<StageRow>>> = {}): StageRow[] =>
  UNITS.map(([key, unit]) => ({
    key, label: STAGE_LABELS[key], state: "todo", done: 0, total: 0, unit, seconds: 0, eta: null, ...over[key],
  }));

const done = (total: number, seconds = 10): Partial<StageRow> => ({ state: "done", done: total, total, seconds });

const item = (over: Partial<JobItem> = {}): JobItem => ({
  videoId: "aaaaaaaaaaa", title: "A talk", channel: "Someone", duration: 10506, thumb: null, status: "running",
  position: null, stopAt: null, settings: null, finalUntil: null, stage: "voice_lines", eta: 7800, updatedAt: 1,
  createdAt: 1, bytes: 1, output: null, slept: null, error: null, expired: false, stages: [], elapsed: 0, coverage: null,
  report: null, ...over,
});

const output = (over: Partial<Output> = {}): Output => ({
  path: "/Users/me/Movies/Maata/A talk (Telugu).mp4", bytes: 3.1e9, kind: "whole", at: 0,
  loudness: { I: -16, TP: -1.2, LRA: 6 }, warning: null, missing: false, ...over,
});

describe("the stepper's groups", () => {
  it("are the design's ten, covering every engine stage once, in order", () => {
    expect(STAGE_GROUPS.map((g) => g.label)).toEqual([
      "Download", "Speakers", "Transcript", "Voices & brief", "Background sound", "Translation", "Telugu speech", "Finishing",
      "Video download", "Saving the video",
    ]);
    expect(STAGE_GROUPS.flatMap((g) => g.keys)).toEqual(UNITS.map(([k]) => k));
  });

  it("show each group's state, count, time and ETA; side-by-side stages count as their longest", () => {
    const rows = groupRows(stages({
      fetch: done(162.4, 80), speakers: done(1, 400), transcript: done(10506, 900), units: done(1900, 1),
      voices: { state: "running", done: 1, total: 2, seconds: 60, eta: 90 }, brief: { state: "running", done: 1, total: 4, seconds: 70, eta: 200 },
    }));
    const by = Object.fromEntries(rows.map((r) => [r.label, r]));
    expect(by.Download).toMatchObject({ state: "done", count: "162 MB", seconds: 80, eta: null });
    expect(by.Speakers).toMatchObject({ state: "done", count: "" });
    expect(by.Transcript).toMatchObject({ state: "done", count: "2:55:06", seconds: 900 });
    expect(by["Voices & brief"]).toMatchObject({ state: "running", count: "1 / 2 speakers", share: 0.5, seconds: 70, eta: 200,
      indeterminate: false });
    expect(by.Translation).toMatchObject({ state: "todo", count: "", eta: null });
  });

  it("move a bar with no count while pyannote clusters, and for a download of unknown size", () => {
    const scan = groupRows(stages({ speakers: { state: "running", done: 0.4, total: 1, eta: 300 } }))[1]!;
    expect(scan).toMatchObject({ count: "40 %", share: 0.4, indeterminate: false, note: "" });
    const clustering = groupRows(stages({ speakers: { state: "running", done: 0.9, total: 1 } }))[1]!;
    expect(clustering).toMatchObject({ count: "", share: null, indeterminate: true, note: CLUSTERING });
    const unknown = groupRows(stages({ video: { state: "running", done: 12, total: 0 } }))[8]!;
    expect(unknown).toMatchObject({ count: "12 MB", indeterminate: true });
  });

  it("say what a stage is doing besides counting (the engine's extra)", () => {
    const rows = groupRows(stages({
      voice_lines: { state: "running", done: 0, total: 9400, extra: "Waiting for the background sound" },
      export: { state: "running", done: 10506, total: 10506, extra: "Finishing the file…" },
    }));
    expect(rows[6]!.note).toBe("Waiting for the background sound");
    expect(rows[9]!.note).toBe("Finishing the file…");
  });

  it("show a failed stage as failed", () => {
    expect(groupRows(stages({ video: { state: "failed", done: 3, total: 300 } }))[8]!.state).toBe("failed");
  });

  it("count each unit as people read it", () => {
    const row = (unit: string, done: number, total: number, state: StageRow["state"] = "running") =>
      countText({ key: "fetch", label: "", state, done, total, unit, seconds: 0, eta: null });
    expect(row("MB", 52.4, 162.4)).toBe("52 / 162 MB");
    expect(row("video s", 750, 10506)).toBe("12:30 / 2:55:06");
    expect(row("speech s", 3600, 9400)).toBe("1:00:00 / 2:36:40");
    expect(row("lines", 120, 1900)).toBe("120 / 1,900 lines");
    expect(row("lines", 1900, 1900, "done")).toBe("1,900 lines");
    expect(row("speakers", 1, 1, "done")).toBe("1 speaker");
    expect(row("parts", 2, 4)).toBe("2 / 4 parts");
    expect(row("lines", 0, 0, "todo")).toBe("");
  });
});

describe("chips", () => {
  const now = Date.UTC(2026, 9, 3, 12, 0);

  it("say where a running job is and what it has left", () => {
    const j = live(item(), {
      videoId: "aaaaaaaaaaa", status: "running", stage: "voice_lines", eta: 7800, elapsed: 3600, position: null, stopAt: null,
      finalUntil: null, output: null, slept: null, coverage: null, error: null,
      stages: stages({ voice_lines: { state: "running", done: 4042, total: 9400 } }),
    });
    expect(chip(j, now)).toEqual({ text: "Dubbing · Telugu speech 43 % · about 2 h 10 min left", tone: "run" });
    expect(chip(item({ stage: "speakers", eta: null }), now).text).toBe("Dubbing · Speakers"); // no render of it yet
  });

  it("give the queue position", () => {
    expect(chip(item({ status: "queued", position: 2 }), now)).toEqual({ text: "Queued · 2nd in line", tone: "wait" });
    expect(chip(item({ status: "queued", position: 1 }), now).text).toBe("Queued · next in line");
  });

  it("say where it paused, and when Maata was closed", () => {
    expect(chip(item({ status: "paused", stage: "translate" }), now).text).toBe("Paused at Translation");
    expect(chip(item({ status: "interrupted" }), now).text).toBe("Continuing after Maata was closed");
  });

  it("say the Mac slept, while that is news", () => {
    const resumed = now / 1000 - 60;
    const slept = { at: resumed - 3600, resumed };
    expect(chip(item({ slept }), now)).toEqual({ text: `Paused while the Mac slept · resumed ${fmtReset(resumed, new Date(now))}`, tone: "wait" });
    expect(chip(item({ slept }), now + (SLEEP_NOTE_FOR + 120) * 1000).text).toMatch(/^Dubbing/);
  });

  it("say when Codex holds a job back, and until when", () => {
    const resets = now / 1000 + 6 * 3600;
    expect(chip(item({ status: "waiting" }), now, resets).text).toBe(`Waiting for Codex · resets ${fmtReset(resets, new Date(now))}`);
    expect(chip(item({ status: "waiting" }), now).text).toBe("Waiting for Codex");
  });

  it("give a done job's length and size, a preview's length, a missing file, a failure and an expired job", () => {
    expect(chip(item({ status: "done", output: output() }), now)).toEqual({ text: "Done · 2:55:06 · 3.1 GB", tone: "ok" });
    expect(chip(item({ status: "done", stopAt: 900, output: output({ kind: "preview" }) }), now).text).toBe("Preview done · first 15 min");
    expect(chip(item({ status: "done", output: output({ missing: true }) }), now)).toEqual({ text: "File moved or deleted", tone: "warn" });
    expect(chip(item({ status: "failed", error: "YouTube refused the video." }), now))
      .toEqual({ text: "Failed · YouTube refused the video.", tone: "danger" });
    expect(chip(item({ status: "done", expired: true }), now).text).toBe("Expired · Dub again to re-voice");
    expect(chip(item({ status: "queued", position: 1, expired: true }), now).text).toBe("Queued · next in line"); // dubbed again
  });
});

describe("queue position, ETA and elapsed", () => {
  it("read as people say them", () => {
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101].map(positionText)).toEqual([
      "next in line", "2nd in line", "3rd in line", "4th in line", "11th in line", "12th in line", "13th in line",
      "21st in line", "22nd in line", "23rd in line", "101st in line",
    ]);
    expect(etaText(7800)).toBe("about 2 h 10 min left");
    expect(etaText(null)).toBe("");
    expect(etaText(25 * 60)).toBe("about 25 min left");
    expect(elapsedText(3900)).toBe("1 h 5 min so far");
    expect(elapsedText(0)).toBe("");
    expect(elapsedText(3 * 3600 + 600, true)).toBe("took 3 h 10 min");
  });
});

describe("New dub's estimate line", () => {
  const est = { seconds: [3 * 3600, 6.4 * 3600] as [number, number], claudeCalls: [310, 360] as [number, number],
    preview: { seconds: [30 * 60, 70 * 60] as [number, number], claudeCalls: [30, 35] as [number, number] }, ahead: null };

  it("gives the time, the Codex calls and where it saves", () => {
    expect(estimateLine(est, { stopAt: null, outputDir: "/Users/me/Movies/Maata", ahead: 0, claude: true }))
      .toBe("About 3–6½ h on this Mac · about 310–360 Codex calls · saves to Movies ▸ Maata");
    expect(estimateLine(est, { stopAt: 900, outputDir: "/Users/me/Movies/Maata", ahead: 0, claude: true }))
      .toBe("About 30–70 min on this Mac · about 30–35 Codex calls · saves to Movies ▸ Maata");
  });

  it("adds the jobs ahead when it would be queued, and leaves Codex out where the engine doesn't use it", () => {
    expect(estimateLine({ ...est, ahead: 7200 }, { stopAt: null, outputDir: null, ahead: 1, claude: false }))
      .toBe("About 3–6½ h on this Mac · after the 1 job ahead (about 2 h)");
    expect(estimateLine({ ...est, ahead: 4 * 3600 + 1200 }, { stopAt: null, outputDir: null, ahead: 2, claude: false }))
      .toBe("About 3–6½ h on this Mac · after the 2 jobs ahead (about 4 h 20 min)");
  });

  it("counts the jobs ahead: the running one and those queued, never the video itself", () => {
    const jobs = [
      { videoId: "a", status: "running" as const, position: null }, { videoId: "b", status: "queued" as const, position: 1 },
      { videoId: "c", status: "interrupted" as const, position: 2 }, { videoId: "d", status: "done" as const, position: null },
      { videoId: "e", status: "paused" as const, position: null },
    ];
    expect(jobsAhead(jobs, "x")).toBe(3);
    expect(jobsAhead(jobs, "b")).toBe(2);
    expect(jobsAhead([], "x")).toBe(0);
  });

  it("names the output folder from the home folder", () => {
    expect(folderText("/Users/me/Movies/Maata")).toBe("Movies ▸ Maata");
    expect(folderText("/home/me/Videos")).toBe("Videos");
    expect(folderText("/Volumes/Big/Dubs")).toBe("Volumes ▸ Big ▸ Dubs");
    expect(folderText("/private/tmp/scratch/demo/out")).toBe("… ▸ demo ▸ out");
  });
});

describe("the Done card and the speaker check", () => {
  it("give the coverage classes as shares of the lines classed", () => {
    expect(coverageShares({ C: 90, m: 8, P: 2, E: 0, otherTier: 3, unreviewed: 1, skipped: 2, lines: 106 })).toEqual([
      { key: "C", label: "Complete", share: 0.9 }, { key: "m", label: "Minor changes", share: 0.08 }, { key: "P", label: "Partly", share: 0.02 },
    ]);
    expect(coverageShares(null)).toEqual([]);
    expect(coverageShares({ C: 0, m: 0, P: 0, E: 0, otherTier: 0, unreviewed: 0, skipped: 0, lines: 0 })).toEqual([]);
  });

  const running = (over: Partial<Record<StageKey, Partial<StageRow>>>): Render => ({
    videoId: "a", status: "running", stage: "transcript", stages: stages(over), stopAt: null, finalUntil: null, eta: 1,
    elapsed: 1, position: null, output: null, slept: null, coverage: null, error: null,
  });

  it("say a speaker change is free until the Telugu speech starts, and what it costs after", () => {
    const early = running({ speakers: done(1), transcript: { state: "running", eta: 600 }, units: { eta: 1 }, voices: { eta: 120 },
      brief: { eta: 300 }, separate: { eta: 299 } });
    expect(untilVoicing(early)).toBe(1200);
    expect(speakerChangeCost(early)).toBe("Free until the background sound is done, in about 20 min.");
    expect(voiceSwitchCost(early, "Speaker 2")).toBe("Free now: nothing is voiced yet.");
    const late = running({ voice_lines: { state: "running", done: 30, total: 9400 } });
    expect(speakerChangeCost(late)).toBe("Lines whose speaker changes are translated and voiced again.");
    expect(voiceSwitchCost(late, "Speaker 2")).toBe("Speaker 2's lines are voiced again.");
    expect(speakerChangeCost({ ...early, status: "paused" })).toBe("Free now: nothing is voiced yet.");
  });

  it("say how many speakers were found and what was merged", () => {
    const sp = (id: string) => ({ id, label: `Speaker ${id.slice(1)}`, talkSeconds: 1, share: 0.5, firstAt: 0, turns: 1, activity: [] });
    const f = { videoId: "a", mode: "auto" as const, fresh: true, freeFor: 0, bounds: null, speakers: [sp("S1"), sp("S2")],
      merged: [{ from: "S3", into: "S1", why: "talk", talkSeconds: 9 }, { from: "S4", into: "S1", why: "talk", talkSeconds: 4 },
        { from: "S5", into: "S2", why: "same", talkSeconds: 40 }] };
    expect(foundText(f)).toEqual({ title: "Found 2 speakers", merges: ["2 short voices merged into Speaker 1", "1 matching voice merged into Speaker 2"] });
    expect(foundText({ ...f, speakers: [sp("S1")], merged: [] })).toEqual({ title: "Found 1 speaker", merges: [] });
  });
});
