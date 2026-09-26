/**
 * SyncEngine end to end, against stand-ins: the fake AudioContext's clock is the (fake) wall clock,
 * and the fake video does what App does on the player's events (the clock follows the video, and
 * "playing" starts the dub). The pause event arrives at once, as the demo player sends it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoClock } from "./clock";
import { SyncEngine, type VideoControl } from "./sync";
import type { DubUnit, Edit } from "./types";

class FakeBuffer {
  id = -1;
  constructor(readonly length: number, readonly sampleRate: number) {}
  get duration(): number {
    return this.length / this.sampleRate;
  }
  copyToChannel(data: Float32Array): void {
    this.id = data[0] ?? -1; // the rig fills a unit's samples with its id
  }
}

class FakeSource {
  buffer: FakeBuffer | null = null;
  onended: (() => void) | null = null;
  when = 0;
  offset = 0;
  stopped = false;
  constructor(private ctx: FakeContext) {}
  connect(): void {}
  start(when: number, offset: number): void {
    this.when = when;
    this.offset = offset;
    this.ctx.started.push(this);
  }
  stop(): void {
    this.stopped = true;
  }
}

class FakeContext {
  static last: FakeContext;
  started: FakeSource[] = [];
  outputLatency = 0;
  baseLatency = 0;
  destination = {};
  constructor() {
    FakeContext.last = this;
  }
  get currentTime(): number {
    return performance.now() / 1000;
  }
  createGain() {
    return { gain: { setTargetAtTime() {} }, connect() {} };
  }
  createBuffer(_channels: number, length: number, sampleRate: number) {
    return new FakeBuffer(length, sampleRate);
  }
  createBufferSource() {
    return new FakeSource(this);
  }
  resume() {
    return Promise.resolve();
  }
  close() {
    return Promise.resolve();
  }
}

const unit = (id: number, start: number, end: number, audioWall: number, edits: Edit[] = []): DubUnit => ({
  id, speaker: "S1", start, end, budget: end - start, audioRate: 1, audioWall, voice: "cloned",
  source: "", telugu: "", units: 1, edits,
});
const freeze = (at: number, added: number): Edit => ({ kind: "freeze", at, span: 0, rate: 1, added });

function rig(rates = [0.5, 0.75, 0.8, 1, 1.25, 1.5]) {
  const clock = new VideoClock();
  const calls: string[] = [];
  let sync!: SyncEngine;
  const video: VideoControl = {
    pause: () => { calls.push("pause"); clock.setPlaying(false); },
    play: () => { calls.push("play"); clock.setPlaying(true); sync.start(); },
    setRate: (r) => { calls.push(`rate ${r}`); clock.setRate(r); },
    rates: () => rates,
  };
  sync = new SyncEngine(clock, video);
  const ctx = FakeContext.last;
  return {
    clock, sync, video, calls,
    /** A unit with `seconds` of audio. */
    add(u: DubUnit, seconds: number) {
      sync.addUnit(u);
      sync.addAudio(u.id, 1000, new Float32Array(Math.round(seconds * 1000)).fill(u.id));
    },
    /** Each start of unit `id`'s audio, in order. */
    starts: (id: number) => ctx.started.filter((s) => s.buffer?.id === id),
    now: () => ctx.currentTime,
  };
}

const wait = (s: number) => vi.advanceTimersByTime(Math.round(s * 1000));

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "performance"] });
  vi.stubGlobal("window", globalThis);
  vi.stubGlobal("AudioContext", FakeContext);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("SyncEngine freezes", () => {
  it("holds the rest of a freeze after a pause mid-hold, and the cut line finishes in it", () => {
    const r = rig();
    r.add(unit(1, 10, 12, 3, [freeze(12, 1)]), 3); // the Telugu outruns its slot by 1 s
    r.add(unit(2, 12, 14, 2), 2);
    r.clock.setPlaying(true, 9.5);
    r.sync.start();
    const t0 = r.now();
    wait(2.6); // line 1 started at +0.5; the freeze holds the video at 12 from +2.5 to +3.5
    expect(r.starts(1)[0]!.when).toBeCloseTo(t0 + 0.5, 2);
    expect(r.calls).toEqual(["pause"]);
    expect(r.sync.freezing).toBe(true);

    wait(0.4);
    expect(r.sync.halt()).toBe(true); // the user pauses with 0.5 s of the hold left
    wait(5);
    expect(r.calls).toEqual(["pause"]);

    r.video.play();
    expect(r.calls).toEqual(["pause", "play", "pause"]); // the frame holds again
    const resumed = r.starts(1)[1]!;
    expect(resumed.offset).toBeCloseTo(2.5, 2); // from where the pause cut it
    expect(r.starts(2)).toHaveLength(0); // the next line waits for the hold
    wait(0.6);
    expect(r.calls.at(-1)).toBe("play");
    expect(r.starts(2)).toHaveLength(1);
    expect(r.starts(2)[0]!.when).toBeGreaterThanOrEqual(resumed.when + 0.5 - 0.05);
  });

  it("reports a seek that cut a hold short, and carryOn rolls the video on", () => {
    const r = rig();
    r.add(unit(1, 10, 12, 3, [freeze(12, 1)]), 3);
    r.clock.setPlaying(true, 9.5);
    r.sync.start();
    wait(2.8);
    expect(r.sync.freezing).toBe(true);
    expect(r.sync.seek()).toBe(true);
    r.clock.anchor(40);
    wait(3);
    expect(r.calls).toEqual(["pause"]); // the hold's own resume went with it: the video would sit paused
    r.sync.carryOn(true);
    expect(r.calls).toEqual(["pause", "play"]);
    expect(r.clock.playing).toBe(true);
    expect(r.sync.freezing).toBe(false);
  });

  it("holds the rest of a freeze at the new speed after a speed change mid-hold", () => {
    const r = rig();
    r.add(unit(1, 10, 12, 3, [freeze(12, 1)]), 3);
    r.clock.setPlaying(true, 9.5);
    r.sync.start();
    wait(3); // 0.5 video s of the hold left
    const held = r.sync.halt();
    expect(held).toBe(true);
    r.sync.speed = 1.25;
    r.video.setRate(1.25);
    r.sync.carryOn(held);
    expect(r.sync.freezing).toBe(true);
    expect(r.starts(1)).toHaveLength(2); // the cut line finishes during the hold
    wait(0.35); // 0.5 video s at 1.25x is 0.4 wall s
    expect(r.calls.at(-1)).toBe("pause");
    wait(0.1);
    expect(r.calls.at(-1)).toBe("play");
    expect(r.clock.playing).toBe(true);
  });

  it("doesn't replay a line that ran out while the hold kept the video inside it", () => {
    const r = rig();
    r.add(unit(1, 10, 12, 3, [freeze(12, 1)]), 3);
    r.clock.setPlaying(true, 10);
    r.sync.start();
    wait(2.99); // the hold ends at +3.0, when the line's audio runs out
    r.starts(1)[0]!.onended?.();
    wait(0.5);
    expect(r.clock.playing).toBe(true); // the video rolls on from 12, 2 s into a 3 s line
    expect(r.starts(1)).toHaveLength(1);
    // A seek back replays it.
    r.sync.seek();
    r.clock.anchor(10);
    r.sync.start();
    expect(r.starts(1)).toHaveLength(2);
  });
});

describe("SyncEngine halts and eviction", () => {
  it("resumes a line after a buffering stall from what was heard, not from the stalled video", () => {
    const r = rig();
    r.add(unit(1, 10, 14, 4), 4);
    r.clock.setPlaying(true, 10);
    r.sync.start();
    expect(r.starts(1)[0]!.offset).toBe(0);
    wait(1.5);
    // The video stalled at 11.2 while the clock ran on to 11.5; the line stops with it.
    r.clock.setPlaying(false, 11.2);
    expect(r.sync.halt()).toBe(false); // no hold to cut short
    expect(r.starts(1)[0]!.stopped).toBe(true);
    wait(2);
    r.clock.setPlaying(true, 11.2);
    r.sync.start();
    expect(r.starts(1)).toHaveLength(2);
    expect(r.starts(1)[1]!.offset).toBeCloseTo(1.5, 2); // not 1.2: never repeat what was heard
  });

  it("keeps a sounding line's audio when evicting", () => {
    const r = rig();
    r.add(unit(1, 10, 14, 4), 4);
    r.add(unit(2, 20, 22, 2), 2);
    r.add(unit(3, 500, 502, 2), 2);
    r.clock.setPlaying(true, 10);
    r.sync.start();
    expect(r.sync.evict(100, 200).map((s) => s.id).sort()).toEqual([2, 3]);
    r.sync.seek();
    r.clock.anchor(20);
    r.sync.start();
    expect(r.starts(2)).toHaveLength(0); // its audio is gone until the engine re-sends it
  });

  it("drops a line's audio the engine now holds back, unless it is sounding, and plays it once sent again", () => {
    const r = rig();
    r.add(unit(1, 10, 14, 4), 4);
    r.add(unit(2, 20, 22, 2), 2);
    r.clock.setPlaying(true, 10);
    r.sync.start();
    r.sync.dropAudio(1); // sounding: it finishes
    r.sync.dropAudio(2);
    wait(10.5);
    expect(r.starts(1)).toHaveLength(1);
    expect(r.starts(2)).toHaveLength(0); // no audio at its time
    r.sync.seek();
    r.clock.anchor(20);
    r.sync.addAudio(2, 1000, new Float32Array(2000).fill(2)); // the engine sent it when asked
    r.sync.start();
    expect(r.starts(2)).toHaveLength(1);
  });
});

describe("SyncEngine slow-downs", () => {
  it("starts the line after a slow-down when the slowed video reaches it", () => {
    const r = rig([0.5, 0.75, 0.8, 1, 1.25]);
    const slow: Edit = { kind: "slow", at: 10, span: 2, rate: 0.8, added: 0.5 };
    r.add(unit(1, 10, 12, 2.5, [slow]), 2.5);
    r.add(unit(2, 12, 14, 2), 2);
    r.clock.setPlaying(true, 9.9);
    r.sync.start();
    const t0 = r.now();
    wait(3);
    expect(r.calls).toEqual(["rate 0.8", "rate 1"]);
    // The slow-down starts at +0.1 (video 10) and runs 2 video s at 0.8x: video 12 is at +2.6, not +2.1.
    expect(r.starts(2)).toHaveLength(1);
    expect(r.starts(2)[0]!.when).toBeCloseTo(t0 + 2.6, 1);
  });
});
