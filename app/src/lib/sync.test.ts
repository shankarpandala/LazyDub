import { describe, expect, it } from "vitest";
import { VideoClock } from "./clock";
import { editKey, freezeBlock, outside, planStart, rateTo, unitSpan } from "./sync";
import type { DubUnit, Edit } from "./types";
import { decodeAudioFrame } from "./engine";
import { fmtTime, mergeRanges, percentile } from "./format";

describe("VideoClock", () => {
  it("extrapolates while playing and holds while paused", () => {
    const c = new VideoClock();
    c.setPlaying(true, 10, 0);
    expect(c.now(1000)).toBeCloseTo(11);
    c.setRate(1.5, 1000);
    expect(c.now(2000)).toBeCloseTo(12.5);
    c.setPlaying(false, undefined, 2000);
    expect(c.now(9000)).toBeCloseTo(12.5);
  });
  it("slews small errors and hard-resyncs large ones", () => {
    const c = new VideoClock();
    c.setPlaying(true, 0, 0);
    c.observe(1.1, 1000); // 100 ms error: slew
    expect(c.now(1000)).toBeCloseTo(1.008, 3);
    c.observe(5, 1000); // 4 s error: jump
    expect(c.now(1000)).toBe(5);
  });
});

describe("planStart", () => {
  it("schedules exactly at s_u, compensating output latency", () => {
    const d = planStart(12, 3, 11, 100, 1, 0.02);
    expect(d).toEqual({ when: 100 + 1 - 0.02, offset: 0 });
  });
  it("accounts for user speed", () => {
    expect(planStart(12, 3, 11, 100, 2, 0)!.when).toBeCloseTo(100.5);
  });
  it("joins mid-line after a seek, but not at the very end", () => {
    expect(planStart(10, 3, 11, 100, 1, 0)).toEqual({ when: 100, offset: 1 });
    expect(planStart(10, 3, 12.99, 100, 1, 0)).toBeNull();
  });
  it("waits beyond the horizon or behind a pending freeze", () => {
    expect(planStart(20, 3, 11, 100, 1, 0)).toBeNull();
    expect(planStart(12, 3, 11, 100, 1, 0, 1.5, 11.5)).toBeNull();
  });
  it("holds a line that starts exactly at a freeze until the freeze ends", () => {
    expect(planStart(12, 3, 11.99, 100, 1, 0, 1.5, 12)).toBeNull();
    expect(planStart(11.9, 3, 11.8, 100, 1, 0, 1.5, 12)).not.toBeNull();
    expect(planStart(12, 3, 12, 100, 1, 0, 1.5, Infinity)).toEqual({ when: 100, offset: 0 });
  });
  it("resumes a cut-off line where it was cut, never repeating what was heard", () => {
    // A freeze let the line run 2.5 s while the video sat at +1.8 s: resume at 2.5, not 1.8.
    expect(planStart(10, 4, 11.8, 100, 1, 0, 1.5, Infinity, 2.5)).toEqual({ when: 100, offset: 2.5 });
    // The video is further on than what was heard: follow the video.
    expect(planStart(10, 4, 12, 100, 1, 0, 1.5, Infinity, 1)).toEqual({ when: 100, offset: 2 });
    // Everything was heard: nothing to play.
    expect(planStart(10, 4, 11, 100, 1, 0, 1.5, Infinity, 3.97)).toBeNull();
  });
});

describe("rateTo", () => {
  const slow = { rate: 0.75, until: 12 };
  it("is the user speed with no slow-down in progress", () => {
    expect(rateTo(20, 10, 1.25, null)).toBe(1.25);
    expect(rateTo(20, 12, 1, slow)).toBe(1); // the slow-down is over
  });
  it("is the slowed rate up to the slow-down's end, so a line there isn't scheduled early", () => {
    expect(rateTo(12, 10.5, 1, slow)).toBe(0.75);
    // 1.5 video s at 0.75x is 2 wall s: beyond the 1.5 s horizon, not inside it
    expect(planStart(12, 3, 10.5, 100, rateTo(12, 10.5, 1, slow), 0)).toBeNull();
    expect(planStart(12, 3, 10.875, 100, rateTo(12, 10.875, 1, slow), 0)!.when).toBeCloseTo(101.5);
  });
  it("blends both rates for a line past the slow-down's end", () => {
    // 1 video s slowed (1.333 wall s), then 1 video s at 1x (1 wall s)
    expect(rateTo(13, 11, 1, slow)).toBeCloseTo(2 / (1 / 0.75 + 1));
  });
});

describe("freezeBlock", () => {
  const edits: Edit[] = [
    { kind: "slow", at: 5, span: 2, rate: 0.9, added: 0.2 },
    { kind: "freeze", at: 12, span: 0, rate: 1, added: 0.8 },
    { kind: "freeze", at: 30, span: 0, rate: 1, added: 0.5 },
  ];
  it("blocks at the next pending freeze, skipping slow-downs", () => {
    expect(freezeBlock(edits, new Set(), 4, null)).toBe(12);
  });
  it("blocks at the active freeze even though it is already done", () => {
    const done = new Set([editKey(edits[1]!)]);
    expect(freezeBlock(edits, done, 12, 12)).toBe(12);
    expect(freezeBlock(edits, done, 12, null)).toBe(30);
  });
  it("is open when no freeze lies ahead", () => {
    expect(freezeBlock(edits, new Set(), 31, null)).toBe(Infinity);
  });
});

describe("eviction", () => {
  const unit = (id: number, start: number, end: number, audioWall: number): DubUnit => ({
    id, speaker: "S1", start, end, budget: end - start, audioRate: 1, audioWall, voice: "cloned",
    source: "", telugu: "", units: 1, edits: [],
  });
  it("spans a unit to the end of its audio when the Telugu outruns the slot", () => {
    expect(unitSpan(unit(1, 10, 12, 3.5))).toEqual({ id: 1, start: 10, end: 13.5 });
    expect(unitSpan(unit(2, 10, 12, 1.5))).toEqual({ id: 2, start: 10, end: 12 });
  });
  it("drops only spans wholly outside the window", () => {
    const spans = [unit(1, 0, 5, 5), unit(2, 100, 104, 6), unit(3, 150, 152, 2), unit(4, 900, 903, 3)].map(unitSpan);
    // playhead 225, keep [105, 225 + 600 + 60]: unit 2 still sounds until 106, unit 4 is past the window's far edge
    expect(outside(spans, 105, 885).map((s) => s.id)).toEqual([1, 4]);
  });
});

describe("wire and format helpers", () => {
  it("decodes a binary audio frame (ADR-007)", () => {
    const buf = new ArrayBuffer(16 + 8);
    const v = new DataView(buf);
    v.setUint32(0, 7, true); v.setUint32(4, 24000, true); v.setUint32(8, 2, true);
    new Float32Array(buf, 16, 2).set([0.25, -0.5]);
    const f = decodeAudioFrame(buf);
    expect(f.id).toBe(7); expect(f.sampleRate).toBe(24000); expect([...f.samples]).toEqual([0.25, -0.5]);
  });
  it("formats times, merges ranges, takes percentiles", () => {
    expect(fmtTime(3723)).toBe("1:02:03");
    expect(fmtTime(65)).toBe("1:05");
    expect(mergeRanges([[5, 6], [0, 2], [2.5, 3]], 1)).toEqual([[0, 3], [5, 6]]);
    expect(percentile([1, 2, 3, 4], 50)).toBe(3);
  });
});
