import { describe, expect, it } from "vitest";
import { lagStats } from "./format";

describe("lagStats", () => {
  it("is empty before any line reports a lag", () => {
    expect(lagStats([])).toEqual({ p50: null, p95: null, n: 0, freezes: 0, frozen: 0 });
    expect(lagStats([{}, {}])).toEqual({ p50: null, p95: null, n: 0, freezes: 0, frozen: 0 });
  });

  it("takes signed percentiles, so early starts count as negative", () => {
    const units = [-0.3, -0.1, 0, 0, 0.05, 0.1, 0.1, 0.2, 0.4, 0.9].map((lag) => ({ lag }));
    const s = lagStats(units);
    expect(s.n).toBe(10);
    expect(s.p50).toBe(0.1);
    expect(s.p95).toBe(0.9);
  });

  it("skips lines without a usable lag", () => {
    const s = lagStats([{ lag: 0.2 }, {}, { lag: Number.NaN }, { lag: 0.4 }]);
    expect(s.n).toBe(2);
    expect(s.p50).toBe(0.4);
  });

  it("counts and sums the picture holds", () => {
    const s = lagStats([{ freeze: 0.4 }, { freeze: 0 }, {}, { freeze: 0.25, lag: 0.6 }]);
    expect(s.freezes).toBe(2);
    expect(s.frozen).toBeCloseTo(0.65);
  });
});
