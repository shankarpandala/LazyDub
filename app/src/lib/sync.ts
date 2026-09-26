/**
 * SyncEngine (spec §6.7, ADR-008): plays dub units through Web Audio, scheduled against the
 * extrapolated player clock, and executes the timeline's slow-down and freeze edits.
 *
 * Audio arrives already time-stretched for the §6.6 speed-up and the user's speed, so a unit plays
 * at 1.0× in the audio graph (no pitch change). Every unit is *planned* to start exactly at its
 * video time s_u; the measured error feeds the debug HUD.
 *
 * Nothing new starts while the video clock is stopped (buffering, a freeze, a pause): lines already
 * sounding may finish, but the next one waits for the video. A line that finished is never replayed
 * until a seek, and one cut off mid-way resumes where it was cut, not from the video position (during a
 * freeze's hold too: that is what the hold is for).
 */
import type { VideoClock } from "./clock";
import type { DubUnit, Edit } from "./types";

export type ScheduleDecision = { when: number; offset: number } | null;

/** A line this close before a freeze (s) still counts as at it: it must wait for the freeze to end. */
const FREEZE_TOL = 0.05;

/**
 * When (AudioContext time) and from where (seconds into the buffer) to start a unit.
 * Returns null if the unit is too far ahead, already over, or at/after a freeze that hasn't ended.
 * `rate` is the video's rate on the way to s_u (see `rateTo`); `heard` is how much of the buffer
 * already played (s), which a resume never repeats.
 */
export function planStart(
  start: number, bufferSeconds: number, videoNow: number, ctxNow: number, rate: number,
  outputLatency: number, horizon = 1.5, blockedBefore = Infinity, heard = 0,
): ScheduleDecision {
  const ahead = (start - videoNow) / rate; // wall seconds until s_u
  if (ahead > horizon || start >= blockedBefore - FREEZE_TOL) return null;
  if (ahead >= 0 && heard <= 0) return { when: ctxNow + ahead - outputLatency, offset: 0 };
  const offset = Math.max(-ahead, heard); // joined late (seek/resume): start mid-line
  if (offset >= bufferSeconds - 0.05) return null;
  return { when: ctxNow, offset };
}

export const editKey = (e: Edit): string => `${e.kind}@${e.at}`;

/** A slow-down edit in progress: the video runs at `rate` until it reaches `until`. */
export type Slow = { rate: number; until: number };

/** Mean video rate (video s per wall s) from `v` until the video reaches `t`, through a slow-down in progress. */
export function rateTo(t: number, v: number, speed: number, slow: Slow | null): number {
  if (!slow || v >= slow.until) return speed;
  if (t <= slow.until) return slow.rate;
  return (t - v) / ((slow.until - v) / slow.rate + (t - slow.until) / speed);
}

/** Video time before which lines may start: an active freeze's point, else the next pending freeze's. */
export function freezeBlock(edits: Edit[], done: Set<string>, videoNow: number, activeAt: number | null): number {
  if (activeAt !== null) return activeAt;
  const e = edits.find((x) => x.kind === "freeze" && x.at >= videoNow - 0.02 && !done.has(editKey(x)));
  return e ? e.at : Infinity;
}

export type Span = { id: number; start: number; end: number };

/** Video span a unit's audio can sound in (a Telugu line may outrun its English slot). */
export const unitSpan = (u: DubUnit): Span => ({ id: u.id, start: u.start, end: Math.max(u.end, u.start + u.audioWall) });

/** Spans wholly outside [lo, hi]: their audio can be dropped. */
export function outside(spans: Iterable<Span>, lo: number, hi: number): Span[] {
  const out: Span[] = [];
  for (const s of spans) if (s.end < lo || s.start > hi) out.push(s);
  return out;
}

type Live = { src: AudioBufferSourceNode; ctxStart: number; offset: number; speed: number };
/** A freeze in progress: `until` is the AudioContext time the hold ends (0 while still pending). */
type Freeze = { at: number; key: string | null; until: number };

export type SyncStats = { errorsMs: number[]; lastErrorMs: number | null };

export type VideoControl = { pause(): void; play(): void; setRate(r: number): void; rates(): number[] };

export class SyncEngine {
  readonly ctx: AudioContext;
  private gain: GainNode;
  private units = new Map<number, DubUnit>();
  private buffers = new Map<number, AudioBuffer>();
  private live = new Map<number, Live>();
  /** Lines that played to their end; not replayed until a seek. */
  private played = new Set<number>();
  /** Seconds of a cut-off line already heard, at 1× user speed. */
  private heard = new Map<number, number>();
  private edits: Edit[] = [];
  private editsDirty = false;
  private doneEdits = new Set<string>();
  private timer = 0;
  private timers = new Set<number>();
  private active = false;
  private freeze: Freeze | null = null;
  private resumeFreeze: { at: number; added: number } | null = null;
  private slow: Slow | null = null;
  speed = 1;
  stats: SyncStats = { errorsMs: [], lastErrorMs: null };

  constructor(private clock: VideoClock, private video: VideoControl) {
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
  }

  /** True while a planned freeze-frame holds (or is about to hold) the video; its pause event must not stop audio. */
  get freezing(): boolean {
    return this.freeze !== null;
  }

  setVolume(v: number): void {
    this.gain.gain.setTargetAtTime(v, this.ctx.currentTime, 0.02);
  }

  /**
   * Let the dub sound: call inside a click or key handler. WebKit starts an AudioContext only during a user gesture, and
   * start() often runs later (the video starts itself once enough dub is banked), where resume() is refused silently.
   */
  unlock(): void {
    if (this.ctx.state !== "running") void this.ctx.resume();
  }

  get audioState(): AudioContextState {
    return this.ctx.state;
  }

  /** What the engine log records about playback (App's "diag" message). */
  diag(): { audio: AudioContextState; rate: number; gain: number; buffers: number; live: number; units: number } {
    return {
      audio: this.ctx.state, rate: this.ctx.sampleRate, gain: Math.round(this.gain.gain.value * 1000) / 1000,
      buffers: this.buffers.size, live: this.live.size, units: this.units.size,
    };
  }

  /** Add or replace a unit (the engine may re-send one). */
  addUnit(u: DubUnit): void {
    this.units.set(u.id, u);
    this.editsDirty = true;
  }

  /** Forget a unit's audio (the engine re-sent the unit without it: it holds the audio back until asked for). */
  dropAudio(id: number): void {
    if (this.live.has(id)) return; // sounding now: it finishes
    this.buffers.delete(id);
    this.played.delete(id);
    this.heard.delete(id);
  }

  addAudio(id: number, sampleRate: number, samples: Float32Array): void {
    const old = this.buffers.get(id);
    if (old && old.length === samples.length && old.sampleRate === sampleRate) return; // a re-send of audio we hold
    const buf = this.ctx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
    this.buffers.set(id, buf);
    if (this.live.has(id)) this.stopOne(id); // re-rendered for a new speed; tick rejoins where it was
  }

  start(): void {
    if (this.active) return;
    this.active = true;
    void this.ctx.resume();
    const r = this.resumeFreeze;
    this.resumeFreeze = null;
    if (r && Math.abs(this.clock.now() - r.at) < 0.3) {
      // A pause cut a freeze short: hold the frame for the rest, so the line it protects can finish.
      this.freeze = { at: r.at, key: null, until: 0 };
      this.hold(r.added);
    }
    this.timer = window.setInterval(() => this.tick(), 40);
    this.tick();
  }

  /**
   * Stop audio (user pause, buffering, seek, rate change). A freeze edit does NOT call this.
   * Returns true if it cut short a hold that paused the video: the hold's own resume is cancelled,
   * so a caller that goes on playing must roll the video itself.
   */
  halt(): boolean {
    const f = this.freeze;
    const held = !!f && f.until > 0;
    if (f) {
      if (f.until === 0 && f.key) this.doneEdits.delete(f.key); // never held: run it when the video gets there
      else if (f.until > 0) {
        const left = (f.until - this.ctx.currentTime) * this.speed;
        if (left > 0.05) this.resumeFreeze = { at: f.at, added: left };
      }
    }
    this.freeze = null;
    this.active = false;
    clearInterval(this.timer);
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    if (this.slow) {
      this.slow = null;
      this.video.setRate(this.speed);
    }
    for (const id of [...this.live.keys()]) this.stopOne(id);
    return held;
  }

  /**
   * Carry on playing after a halt (a seek, a speed change). A halt that cut a freeze's hold short
   * cancelled the hold's resume, and the video sits paused: hold the frame for the rest of the hold
   * (after a speed change), or roll the video on now (after a seek, which leaves the freeze behind).
   */
  carryOn(held: boolean): void {
    if (this.clock.playing) this.start();
    else if (held) {
      this.start();
      if (!this.freezing) this.video.play();
    }
  }

  /** Forget what played (a seek). Returns true if it cut short a freeze's hold, as `halt` does. */
  seek(): boolean {
    const held = this.halt();
    this.resumeFreeze = null;
    this.doneEdits.clear();
    this.played.clear();
    this.heard.clear();
    return held;
  }

  /** Forget the current video (a new one is opening). */
  reset(): void {
    this.seek();
    this.units.clear();
    this.buffers.clear();
    this.edits = [];
    this.editsDirty = false;
    this.stats = { errorsMs: [], lastErrorMs: null };
  }

  /**
   * Drop audio outside [lo, hi] to bound webview memory (§6.8). Unit metadata is kept: it is small,
   * and after a seek the engine re-sends only the audio. Returns the spans whose audio went.
   */
  evict(lo: number, hi: number): Span[] {
    const spans: Span[] = [];
    for (const id of this.buffers.keys()) {
      const u = this.units.get(id);
      if (u && !this.live.has(id)) spans.push(unitSpan(u));
    }
    const gone = outside(spans, lo, hi);
    for (const s of gone) {
      this.buffers.delete(s.id);
      this.played.delete(s.id);
      this.heard.delete(s.id);
    }
    for (const k of [...this.doneEdits]) if (Number(k.split("@")[1]) < lo) this.doneEdits.delete(k);
    return gone;
  }

  private stopOne(id: number): void {
    const l = this.live.get(id);
    if (!l) return;
    l.src.onended = null;
    try { l.src.stop(); } catch { /* not started */ }
    this.live.delete(id);
    const t = this.ctx.currentTime - l.ctxStart;
    if (t > 0) this.heard.set(id, Math.max(this.heard.get(id) ?? 0, (l.offset + t) * l.speed));
  }

  private nextEditAfter(v: number): Edit | undefined {
    return this.edits.find((e) => e.at >= v - 0.02 && !this.doneEdits.has(editKey(e)));
  }

  private tick(): void {
    if (!this.active) return;
    if (this.editsDirty) {
      this.edits = [...this.units.values()].flatMap((u) => u.edits).sort((a, b) => a.at - b.at);
      this.editsDirty = false;
    }
    // Buffering, paused or not yet rolling: let sounding lines finish, start nothing against a still clock.
    // A freeze's hold is the exception: a line cut short before the freeze (a pause mid-hold) finishes in it.
    if (!this.clock.playing && !this.freeze?.until) return;
    const v = this.clock.now();
    const ctxNow = this.ctx.currentTime;
    const latency = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    const edit = this.nextEditAfter(v);
    // Wall seconds to video time t: a slow-down in progress runs the video slower than the user speed.
    const wallTo = (t: number) => (t - v) / rateTo(t, v, this.speed, this.slow);

    if (edit && !this.freeze && wallTo(edit.at) <= 0.06) this.execute(edit, wallTo(edit.at));
    const block = freezeBlock(this.edits, this.doneEdits, v, this.freeze ? this.freeze.at : null);

    for (const [id, buf] of this.buffers) {
      if (this.live.has(id) || this.played.has(id)) continue;
      const u = this.units.get(id);
      if (!u) continue;
      const rate = rateTo(u.start, v, this.speed, this.slow);
      const d = planStart(u.start, buf.duration, v, ctxNow, rate, latency, 1.5, block, (this.heard.get(id) ?? 0) / this.speed);
      if (!d) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.gain);
      const at = Math.max(d.when, ctxNow);
      src.start(at, d.offset);
      src.onended = () => {
        this.live.delete(id);
        this.heard.delete(id);
        this.played.add(id);
      };
      this.live.set(id, { src, ctxStart: at, offset: d.offset, speed: this.speed });
      if (d.offset === 0) this.measure(u.start, d.when + latency);
    }
  }

  /** Record planned vs scheduled start: video time at the audible moment minus s_u. */
  private measure(planned: number, audibleCtx: number): void {
    const wallAhead = audibleCtx - this.ctx.currentTime;
    const videoAt = this.clock.now(performance.now() + wallAhead * 1000);
    const errMs = (videoAt - planned) * 1000;
    this.stats.lastErrorMs = errMs;
    this.stats.errorsMs.push(errMs);
    if (this.stats.errorsMs.length > 500) this.stats.errorsMs.shift();
  }

  /** Run `fn` in `s` seconds; halt() cancels it. */
  private later(fn: () => void, s: number): void {
    if (s <= 0.004) return fn();
    const t = window.setTimeout(() => {
      this.timers.delete(t);
      fn();
    }, s * 1000);
    this.timers.add(t);
  }

  private execute(e: Edit, inSeconds: number): void {
    this.doneEdits.add(editKey(e));
    if (e.kind === "freeze") {
      // Pending counts as frozen: nothing at or after e.at may start before the hold ends.
      this.freeze = { at: e.at, key: editKey(e), until: 0 };
      this.later(() => this.hold(e.added), inSeconds);
    } else {
      this.later(() => {
        const rates = this.video.rates();
        const want = e.rate * this.speed;
        const r = rates.filter((x) => x >= want - 1e-3).sort((a, b) => a - b)[0] ?? this.speed;
        this.slow = { rate: r, until: this.clock.now() + e.span };
        this.video.setRate(r);
        this.later(() => {
          this.slow = null;
          this.video.setRate(this.speed);
        }, e.span / r);
      }, inSeconds);
    }
  }

  /** Hold the frame for `added` video seconds, then roll on. */
  private hold(added: number): void {
    const wall = added / this.speed;
    if (this.freeze) this.freeze.until = this.ctx.currentTime + wall;
    this.video.pause();
    this.later(() => {
      this.freeze = null;
      this.video.play();
    }, wall);
  }

  get scheduledCount(): number {
    return this.live.size;
  }

  dispose(): void {
    this.halt();
    void this.ctx.close();
  }
}
