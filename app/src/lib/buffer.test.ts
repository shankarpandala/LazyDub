import { describe, expect, it } from "vitest";
import {
  ASK_AHEAD, END_SLACK, MAX_ASKS, MissingAudio, REASK_MS, bufferStep, clipLead, contiguousLead, etaSeconds, fmtEta,
  needSeconds, progress, startDecision, type BufferSnapshot,
} from "./buffer";

describe("needSeconds", () => {
  it("defaults to the user's prepare-ahead when the engine has no opinion", () => {
    expect(needSeconds({ remaining: 3600, prepareAhead: 300, targetLead: 0 })).toBe(300);
  });
  it("takes the engine's target when it asks for more (a slow engine)", () => {
    expect(needSeconds({ remaining: 3600, prepareAhead: 300, targetLead: 840 })).toBe(840);
    expect(needSeconds({ remaining: 3600, prepareAhead: 300, targetLead: 60 })).toBe(300);
  });
  it("never asks for more than is left of the video, less the end slack", () => {
    expect(needSeconds({ remaining: 95, prepareAhead: 300, targetLead: 0 })).toBe(95 - END_SLACK);
    expect(needSeconds({ remaining: 0, prepareAhead: 300, targetLead: 120 })).toBe(0);
    expect(needSeconds({ remaining: Infinity, prepareAhead: 120, targetLead: 0 })).toBe(120);
  });
  it("caps the user's setting by what the engine's look-ahead can bank", () => {
    expect(needSeconds({ remaining: 3600, prepareAhead: 600, targetLead: 0, lookahead: 300 })).toBe(240);
    expect(needSeconds({ remaining: 3600, prepareAhead: 300, targetLead: 0, lookahead: 600 })).toBe(300);
  });
  it("doesn't cap the engine's target by the look-ahead: the engine works past it to bank it", () => {
    expect(needSeconds({ remaining: 3600, prepareAhead: 600, targetLead: 500, lookahead: 300 })).toBe(500);
    expect(needSeconds({ remaining: 3600, prepareAhead: 300, targetLead: 400, lookahead: 600 })).toBe(400);
  });
  it("banks what a slow engine needs to play to the end, and says how long that takes (60 min at 0.7x)", () => {
    const targetLead = 3600 * (1 - 0.7) + 30; // pacing.target_lead: 1110 s
    const need = needSeconds({ remaining: 3600, prepareAhead: 300, targetLead, lookahead: 600 });
    expect(need).toBeCloseTo(1110); // not 480 s: that ran dry after 480 / 0.3 = 26.7 min of a 60-min video
    const waiting: BufferSnapshot = {
      wantPlay: true, waiting: true, playing: false, freezing: false, linePlaying: false, lead: 600, need, remaining: 3600,
    };
    expect(bufferStep(waiting)).toBeNull(); // a look-ahead's worth is not enough to start
    expect(bufferStep({ ...waiting, lead: need })).toBe("start");
    expect(etaSeconds(0, need, 0.7)).toBeCloseTo(1110 / 0.7); // the wait: the lead / r of wall time
    // Played from then on, the lead drains at 1 - r and lasts to the end (within the margin).
    expect(need - (3600 - need) * (1 - 0.7)).toBeGreaterThan(0);
  });
});

describe("contiguousLead", () => {
  const ranges: [number, number][] = [[0, 130], [130.1, 200], [400, 700]];
  it("follows the merged ranges from the playhead, bridging rounding gaps", () => {
    expect(contiguousLead(10, ranges, 0)).toBe(190);
    expect(contiguousLead(450, ranges, 0)).toBe(250);
  });
  it("is zero in undubbed video, whatever a stale `until` says", () => {
    expect(contiguousLead(250, ranges, 700)).toBe(0);
    expect(contiguousLead(800, ranges, 900)).toBe(0);
  });
  it("counts a range starting just after the playhead", () => {
    expect(contiguousLead(399.9, ranges, 0)).toBeCloseTo(300.1);
  });
  it("falls back to `until` only when the engine sends no ranges", () => {
    expect(contiguousLead(10, [], 70)).toBe(60);
    expect(contiguousLead(80, [], 70)).toBe(0);
  });
});

describe("clipLead", () => {
  it("stops the lead at audio the UI evicted and is waiting to get back", () => {
    expect(clipLead(100, 300, [[250, 253]])).toBe(150);
    expect(clipLead(100, 300, [[98, 102]])).toBe(0);
    expect(clipLead(100, 300, [[20, 30], [500, 510]])).toBe(300);
  });
  it("only at audio near the playhead: the engine holds the rest and sends it when asked", () => {
    expect(clipLead(100, 900, [[250, 253]], ASK_AHEAD)).toBe(900); // 150 s ahead: asked for in time
    expect(clipLead(100, 900, [[150, 153], [250, 253]], ASK_AHEAD)).toBe(50);
  });
});

describe("MissingAudio", () => {
  const span = (id: number, start: number, end: number) => ({ id, start, end });

  it("tracks evicted spans until their audio comes back", () => {
    const m = new MissingAudio();
    m.add([span(1, 100, 104), span(2, 104, 110), span(3, 300, 303)]);
    expect(m.dirty).toBe(true);
    expect(m.ranges()).toEqual([[100, 110], [300, 303]]);
    expect(m.dirty).toBe(false);
    m.got(2);
    m.got(99); // never missing: no change
    expect(m.ranges()).toEqual([[100, 104], [300, 303]]);
    m.got(1); m.got(3);
    expect(m.size).toBe(0);
  });

  it("doesn't wait on a line a seek can't bring back (seek 8 s into a 12 s line)", () => {
    // The engine re-sends lines starting at or after t - 5 (session._resend_near): 1500 won't come back.
    const m = new MissingAudio();
    m.add([span(1, 1500, 1512), span(2, 1512, 1520), span(3, 1504, 1509), span(4, 1400, 1410)]);
    m.forgetBefore(1508);
    expect(m.ranges()).toEqual([[1400, 1410], [1504, 1509], [1512, 1520]]);
    // A line the seek re-sends still holds the lead at 0 until it arrives; the dropped one never did.
    expect(clipLead(1508, 600, m.ranges())).toBe(0);
    m.got(3);
    expect(clipLead(1508, 600, m.ranges())).toBe(4);
  });

  it("asks for lines by id as the playhead nears them, at most every REASK_MS, and stops waiting after MAX_ASKS", () => {
    const m = new MissingAudio();
    m.add([span(1, 200, 205), span(2, 230, 236)]);
    expect(m.ask(100, 190, 0)).toEqual([]); // beyond the horizon
    expect(m.ask(150, 210, 1000)).toEqual([1]); // never asked; counted as asked now
    expect(m.ask(150, 240, 1000 + REASK_MS - 1)).toEqual([2]); // 1 was just asked
    expect(MAX_ASKS).toBe(2);
    expect(m.ask(150, 210, 1000 + REASK_MS)).toEqual([1]); // its second ask
    m.ranges();
    expect(m.ask(150, 210, 1000 + 2 * REASK_MS)).toEqual([]); // asked enough: no longer waited on
    expect(m.size).toBe(1);
    expect(m.dirty).toBe(true);
  });

  it("counts a seek as an ask for what it re-sends", () => {
    const m = new MissingAudio();
    m.add([span(1, 110, 115), span(2, 900, 905), span(3, 40, 50)]);
    m.asked(100, 600, 5000); // re-sends [95, 700): not unit 2 (too far), unit 3 is behind
    expect(m.ask(100, 1000, 5001)).toEqual([2]); // unit 2 still due; unit 1 was just asked
    expect(m.size).toBe(3);
  });
});

describe("ETA", () => {
  it("is the missing lead over the engine's throughput (the video waits)", () => {
    expect(etaSeconds(130, 300, 2)).toBe(85);
    expect(etaSeconds(0, 300, 0.5)).toBe(600);
    expect(etaSeconds(300, 300, 2)).toBe(0);
  });
  it("is unknown until the engine has measured its throughput", () => {
    expect(etaSeconds(0, 300, 0)).toBeNull();
    expect(etaSeconds(0, 300, Number.NaN)).toBeNull();
  });
  it("reads as a rough human duration", () => {
    expect(fmtEta(4)).toBe("a few seconds left");
    expect(fmtEta(41)).toBe("about 50 s left");
    expect(fmtEta(85)).toBe("about 1 min left");
    expect(fmtEta(600)).toBe("about 10 min left");
    expect(fmtEta(3600 + 20 * 60)).toBe("about 1 h 20 min left");
  });
  it("reports progress as a clamped fraction", () => {
    expect(progress(130, 300)).toBeCloseTo(0.4333, 3);
    expect(progress(500, 300)).toBe(1);
    expect(progress(0, 0)).toBe(1);
  });
});

describe("startDecision", () => {
  it("waits for the full need before the first play and after running dry", () => {
    expect(startDecision(299, 300, true)).toBe("wait");
    expect(startDecision(300, 300, true)).toBe("play");
  });
  it("carries on after a pause or seek with a modest lead", () => {
    expect(startDecision(45, 300, false)).toBe("play");
    expect(startDecision(12, 300, false)).toBe("wait");
    expect(startDecision(9, 9, false)).toBe("play"); // the last few seconds of the video
  });
});

describe("bufferStep", () => {
  const base: BufferSnapshot = {
    wantPlay: true, waiting: false, playing: true, freezing: false, linePlaying: false, lead: 200, need: 300, remaining: 1800,
  };
  it("does nothing unless the user wants to play", () => {
    expect(bufferStep({ ...base, wantPlay: false, lead: 0 })).toBeNull();
  });
  it("keeps playing on a lead well under need: no small-lead stop-and-go", () => {
    expect(bufferStep({ ...base, lead: 5 })).toBeNull();
    expect(bufferStep({ ...base, lead: 0.6 })).toBeNull();
  });
  it("stalls only when the dub truly runs out", () => {
    expect(bufferStep({ ...base, lead: 0.3 })).toBe("stall");
  });
  it("lets a sounding line or a freeze finish before stalling", () => {
    expect(bufferStep({ ...base, lead: 0.3, linePlaying: true })).toBeNull();
    expect(bufferStep({ ...base, lead: 0.3, freezing: true })).toBeNull();
  });
  it("doesn't stall at the very end of the video or while the video isn't running", () => {
    expect(bufferStep({ ...base, lead: 0.2, remaining: 0.3 })).toBeNull();
    // the engine's audio ends a little before YouTube's duration
    expect(bufferStep({ ...base, lead: 0.3, remaining: 1.8 })).toBeNull();
    expect(bufferStep({ ...base, lead: 0, playing: false })).toBeNull();
  });
  it("resumes after running dry only once the full need is banked", () => {
    const waiting = { ...base, waiting: true, playing: false };
    expect(bufferStep({ ...waiting, lead: 30 })).toBeNull();
    expect(bufferStep({ ...waiting, lead: 299.5 })).toBeNull();
    expect(bufferStep({ ...waiting, lead: 300 })).toBe("start");
  });
  it("starts near the end once the rest of the video is ready", () => {
    expect(bufferStep({ ...base, waiting: true, playing: false, lead: 42, need: 42, remaining: 42 })).toBe("start");
  });
  it("never flaps between start and stall near the end", () => {
    for (const remaining of [2.2, 2.4, 3, 10]) {
      for (const lead of [0, 0.2, 0.4]) {
        const need = needSeconds({ remaining, prepareAhead: 300, targetLead: 0 });
        const waiting = { ...base, waiting: true, playing: false, lead, need, remaining };
        if (bufferStep(waiting) === "start") expect(bufferStep({ ...waiting, waiting: false, playing: true })).toBeNull();
      }
    }
  });
});
