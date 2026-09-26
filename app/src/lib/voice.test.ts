import { describe, expect, it } from "vitest";
import {
  CLONE_STRENGTH_OPTIONS, cloneStrengthLabel, clonedLabel, flipPreset, lineMatch, matchSummary, preparedAhead, presetConfirmBody,
  presetLineCounts, presetReason, settlePreset, speakerStatus, voiceNote,
} from "./voice";
import type { DubUnit, Speaker } from "./types";

const speaker = (over: Partial<Speaker> = {}): Speaker => ({
  id: "S1", label: "Speaker 1", voice: "cloned", status: "cloned", referenceSeconds: 10, talkSeconds: 120, usePreset: false, ...over,
});

const unit = (id: number, speakerId: string, voice: DubUnit["voice"]): DubUnit => ({
  id, speaker: speakerId, start: id * 4, end: id * 4 + 3, budget: 3, audioRate: 1, audioWall: 3, voice,
  source: "An example line.", telugu: "ఒక ఉదాహరణ.", units: 8, edits: [],
});

describe("voice match options", () => {
  it("offers the three strengths, closest first, with the engine's ids", () => {
    expect(CLONE_STRENGTH_OPTIONS.map((o) => o.id)).toEqual(["closest", "balanced", "natural"]);
    expect(CLONE_STRENGTH_OPTIONS.map((o) => o.label)).toEqual(["Closest to original", "Balanced", "Most natural Telugu"]);
  });
  it("labels each strength", () => {
    expect(cloneStrengthLabel("closest")).toBe("Closest to original");
    expect(cloneStrengthLabel("balanced")).toBe("Balanced");
    expect(cloneStrengthLabel("natural")).toBe("Most natural Telugu");
  });
  it("recommends none of them: the default is picked by ear in E3", () => {
    for (const o of CLONE_STRENGTH_OPTIONS) expect(`${o.label} ${o.hint}`).not.toMatch(/recommend/i);
  });
});

describe("which voice match a line shows", () => {
  it("names no match when the engine reports none, whatever was requested", () => {
    const u = unit(1, "S1", "cloned");
    expect(lineMatch(u, speaker())).toBeNull();
    expect(clonedLabel(u, speaker())).toBe("Cloned");
    expect(clonedLabel(u, undefined)).toBe("Cloned");
  });
  it("names the match the engine reports for the speaker", () => {
    expect(clonedLabel(unit(1, "S1", "cloned"), speaker({ cloneStrength: "closest" }))).toBe("Cloned · Closest to original");
  });
  it("prefers the match reported for the line itself", () => {
    const u = { ...unit(1, "S1", "cloned"), cloneStrength: "natural" as const };
    expect(lineMatch(u, speaker({ cloneStrength: "closest" }))).toBe("natural");
  });
  it("names no match for a stock-voice line", () => {
    expect(lineMatch({ ...unit(1, "S1", "preset"), cloneStrength: "closest" }, speaker({ cloneStrength: "closest" }))).toBeNull();
  });
});

describe("matchSummary", () => {
  it("only says 'requested' before any voice is cloned", () => {
    const m = matchSummary("balanced", [speaker({ voice: "preset", status: "cloning" })]);
    expect(m).toMatchObject({ mode: "Balanced", note: "requested", warn: false });
    expect(matchSummary("balanced", [])).toMatchObject({ note: "requested" });
  });
  it("says 'not applied yet' when voices are cloned but the engine reports no match for them", () => {
    const m = matchSummary("balanced", [speaker(), speaker({ id: "S2", label: "Speaker 2" })]);
    expect(m).toMatchObject({ mode: "Balanced", note: "requested, not applied yet", warn: true });
  });
  it("claims the match only when the engine reports every cloned voice built with it", () => {
    const both = [speaker({ cloneStrength: "balanced" }), speaker({ id: "S2", cloneStrength: "balanced" })];
    expect(matchSummary("balanced", both)).toMatchObject({ mode: "Balanced", note: "", warn: false });
  });
  it("flags a fallback to another match, or a voice with no report", () => {
    const fellBack = [speaker({ cloneStrength: "balanced" }), speaker({ id: "S2", cloneStrength: "closest" })];
    expect(matchSummary("balanced", fellBack)).toMatchObject({ note: "requested, not used for every voice", warn: true });
    const partial = [speaker({ cloneStrength: "balanced" }), speaker({ id: "S2" })];
    expect(matchSummary("balanced", partial)).toMatchObject({ note: "requested, not used for every voice", warn: true });
  });
  it("ignores speakers the engine hasn't cloned", () => {
    const m = matchSummary("closest", [speaker({ cloneStrength: "closest" }), speaker({ id: "S2", voice: "preset", status: "preset" })]);
    expect(m.note).toBe("");
  });
});

describe("the preset switch", () => {
  it("asks before switching a speaker to the stock voice, and sends nothing yet", () => {
    expect(flipPreset(speaker(), null)).toEqual({ confirming: "S1", send: null });
  });
  it("closes the question on a second click", () => {
    expect(flipPreset(speaker(), "S1")).toEqual({ confirming: null, send: null });
  });
  it("moves the question to the speaker clicked last", () => {
    expect(flipPreset(speaker({ id: "S2" }), "S1")).toEqual({ confirming: "S2", send: null });
  });
  it("switches back to the clone at once, without asking", () => {
    expect(flipPreset(speaker({ usePreset: true }), null)).toEqual({ confirming: null, send: { speaker: "S1", usePreset: false } });
    expect(flipPreset(speaker({ usePreset: true }), "S2")).toEqual({ confirming: null, send: { speaker: "S1", usePreset: false } });
  });
  it("switches only when the listener confirms", () => {
    expect(settlePreset("S1", true)).toEqual({ confirming: null, send: { speaker: "S1", usePreset: true } });
    expect(settlePreset("S1", false)).toEqual({ confirming: null, send: null });
  });
});

describe("when a preset switch is heard", () => {
  // Speaker 1's lines every 4 s; the playhead at 10 s has played lines 1 and 2 (start 4, 8).
  const units = [1, 2, 3, 4, 5].map((i) => unit(i, "S1", "cloned")).concat([unit(6, "S2", "cloned"), unit(7, "S1", "preset")]);

  it("counts a speaker's prepared lines that haven't started, in one voice, and when the last ends", () => {
    expect(preparedAhead(units, "S1", 10, "cloned")).toEqual({ lines: 3, until: 23 });
    expect(preparedAhead(units, "S1", 10, "preset")).toEqual({ lines: 1, until: 31 });
    expect(preparedAhead(units, "S1", 30, "cloned")).toEqual({ lines: 0, until: 0 });
    expect(preparedAhead(units, "S3", 0, "cloned")).toEqual({ lines: 0, until: 0 });
  });
  it("says which prepared lines keep their voice, instead of promising the next line", () => {
    const body = presetConfirmBody("cloned", { lines: 3, until: 623 });
    expect(body).toContain("Lines not yet prepared will be dubbed in a stock Telugu voice");
    expect(body).toContain("The 3 lines already prepared, up to 10:23, keep the voice they were made with.");
    expect(body).not.toMatch(/next lines/);
    expect(presetConfirmBody("cloned", { lines: 1, until: 65 })).toContain("The line already prepared, up to 1:05, keeps the voice it was made with.");
    expect(presetConfirmBody("cloning", { lines: 0, until: 0 })).toContain("even once their clone is ready");
    expect(presetConfirmBody("cloning", { lines: 0, until: 0 })).toContain("Lines already dubbed don't change.");
  });
  it("tells a speaker switched to the preset voice that prepared lines still use the clone", () => {
    expect(voiceNote(speaker({ usePreset: true }), units, 10)).toBe("Stock Telugu voice from 0:23: 3 prepared lines still use the cloned voice");
    expect(voiceNote(speaker({ usePreset: true }), units, 18)).toBe("Stock Telugu voice from 0:23: 1 prepared line still uses the cloned voice");
    expect(voiceNote(speaker({ usePreset: true }), units, 40)).toBe("Stock Telugu voice, not a clone: you switched it on");
  });
  it("tells a speaker switched back that prepared lines still use the preset voice", () => {
    expect(voiceNote(speaker(), units, 10)).toBe("Cloned voice from 0:31: 1 prepared line still uses the preset voice");
    expect(voiceNote(speaker(), units, 40)).toBe("1 line dubbed in the preset voice");
  });
  it("explains a stock-voice fallback, and says nothing for a clean clone", () => {
    expect(voiceNote(speaker({ voice: "preset", status: "preset" }), units, 0)).toBe("Stock Telugu voice, not a clone: this speaker couldn't be cloned yet");
    expect(voiceNote(speaker({ id: "S2", label: "Speaker 2" }), units, 0)).toBeNull();
  });
});

describe("speakerStatus", () => {
  it("shows the listener's preset choice over a finished clone", () => {
    expect(speakerStatus(speaker({ usePreset: true }))).toBe("preset");
  });
  it("shows the engine's status otherwise", () => {
    expect(speakerStatus(speaker())).toBe("cloned");
    expect(speakerStatus(speaker({ status: "cloning" }))).toBe("cloning");
    expect(speakerStatus(speaker({ voice: "preset", status: "preset" }))).toBe("preset");
  });
  it("falls back to the voice kind when the engine sends no status", () => {
    expect(speakerStatus(speaker({ status: undefined as unknown as Speaker["status"] }))).toBe("cloned");
    expect(speakerStatus(speaker({ voice: "preset", status: undefined as unknown as Speaker["status"] }))).toBe("found");
  });
});

describe("presetReason", () => {
  it("is 'chosen' when the listener switched the speaker to the stock voice", () => {
    expect(presetReason(speaker({ usePreset: true }))).toBe("chosen");
  });
  it("is 'fallback' when the engine has no clone for the speaker", () => {
    expect(presetReason(speaker({ voice: "preset", status: "preset" }))).toBe("fallback");
  });
  it("is null while the speaker is cloned or being cloned", () => {
    expect(presetReason(speaker())).toBeNull();
    expect(presetReason(speaker({ status: "cloning", voice: "preset" }))).toBeNull();
    expect(presetReason(speaker({ status: "found", voice: "preset" }))).toBeNull();
  });
});

describe("presetLineCounts", () => {
  it("counts lines dubbed in the stock voice per speaker", () => {
    const units = [unit(1, "S1", "cloned"), unit(2, "S1", "preset"), unit(3, "S2", "preset"), unit(4, "S1", "preset"), unit(5, "S2", "cloned")];
    const counts = presetLineCounts(units);
    expect(counts.get("S1")).toBe(2);
    expect(counts.get("S2")).toBe(1);
    expect(counts.has("S3")).toBe(false);
  });
  it("is empty when every line is cloned", () => {
    expect(presetLineCounts([unit(1, "S1", "cloned")]).size).toBe(0);
    expect(presetLineCounts([]).size).toBe(0);
  });
});
