import { describe, expect, it } from "vitest";
import { VideoClock } from "./clock";
import { planStart } from "./sync";
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
