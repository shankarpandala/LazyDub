import { describe, expect, it } from "vitest";
import { fmtBytes, fmtCount, fmtDuration, fmtSpan, fmtTime, speakerHue } from "./format";

describe("fmtTime", () => {
  it("shows a length with hours only when it has them", () => {
    expect(fmtTime(245)).toBe("4:05");
    expect(fmtTime(10506)).toBe("2:55:06");
    expect(fmtTime(Number.NaN)).toBe("0:00");
  });
});

describe("fmtDuration", () => {
  it("says seconds, then minutes, then hours to the 5 minutes", () => {
    expect(fmtDuration(0)).toBe("1 s");
    expect(fmtDuration(40)).toBe("40 s");
    expect(fmtDuration(61)).toBe("1 min");
    expect(fmtDuration(12 * 60 + 20)).toBe("12 min");
    expect(fmtDuration(3599)).toBe("1 h");
    expect(fmtDuration(2 * 3600 + 9 * 60)).toBe("2 h 10 min");
    expect(fmtDuration(3 * 3600 + 1)).toBe("3 h");
  });
});

describe("fmtSpan", () => {
  it("gives an estimate's range in minutes or half hours", () => {
    expect(fmtSpan(30 * 60, 70 * 60)).toBe("30–70 min");
    expect(fmtSpan(3 * 3600, 6.4 * 3600)).toBe("3–6½ h");
    expect(fmtSpan(50 * 60, 2 * 3600)).toBe("50 min – 2 h");
    expect(fmtSpan(4 * 60, 4 * 60 + 10)).toBe("4 min");
    expect(fmtSpan(20, 50)).toBe("1 min");
  });
});

describe("fmtBytes and fmtCount", () => {
  it("rounds as a file browser would", () => {
    expect(fmtBytes(3.14e9)).toBe("3.1 GB");
    expect(fmtBytes(850.4e6)).toBe("850 MB");
    expect(fmtBytes(12_300)).toBe("12 KB");
    expect(fmtCount(1900)).toBe("1,900");
  });
});

describe("speakerHue", () => {
  it("gives each speaker its own hue, in turn", () => {
    expect(speakerHue("S1")).toBe(24);
    expect(speakerHue("S2")).toBe(330);
    expect(speakerHue("S7")).toBe(24);
  });
});
