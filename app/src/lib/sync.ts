/**
 * SyncEngine (spec §6.7, ADR-008): plays dub units through Web Audio, scheduled against the
 * extrapolated player clock, and executes the timeline's slow-down and freeze edits.
 *
 * Audio arrives already time-stretched for the §6.6 speed-up and the user's speed, so a unit plays
 * at 1.0× in the audio graph (no pitch change). Every unit is *planned* to start exactly at its
 * video time s_u; the measured error feeds the debug HUD.
 */
import type { VideoClock } from "./clock";
import type { DubUnit, Edit } from "./types";

export type ScheduleDecision = { when: number; offset: number } | null;

/**
 * When (AudioContext time) and from where (seconds into the buffer) to start a unit.
 * Returns null if the unit is too far ahead, already over, or an edit lies before it.
 */
export function planStart(
  start: number, bufferSeconds: number, videoNow: number, ctxNow: number, speed: number,
  outputLatency: number, horizon = 1.5, blockedBefore = Infinity,
): ScheduleDecision {
  const ahead = (start - videoNow) / speed; // wall seconds until s_u
  if (ahead > horizon || start > blockedBefore) return null;
  if (ahead >= 0) return { when: ctxNow + ahead - outputLatency, offset: 0 };
  const offset = -ahead; // joined late (seek/resume): start mid-line
  if (offset >= bufferSeconds - 0.05) return null;
  return { when: ctxNow, offset };
}

type Live = { src: AudioBufferSourceNode; planned: number };

export type SyncStats = { errorsMs: number[]; lastErrorMs: number | null };

export class SyncEngine {
  readonly ctx: AudioContext;
  private gain: GainNode;
  private units = new Map<number, DubUnit>();
  private buffers = new Map<number, AudioBuffer>();
  private live = new Map<number, Live>();
  private edits: Edit[] = [];
  private doneEdits = new Set<string>();
  private timer = 0;
  private active = false;
  private freezeTimer = 0;
  speed = 1;
  /** True while a planned freeze-frame holds the video; the player's pause event must not stop audio. */
  freezing = false;
  stats: SyncStats = { errorsMs: [], lastErrorMs: null };

  constructor(
    private clock: VideoClock,
    private video: { pause(): void; play(): void; setRate(r: number): void; rates(): number[] },
  ) {
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
  }

  setVolume(v: number): void {
    this.gain.gain.setTargetAtTime(v, this.ctx.currentTime, 0.02);
  }

  addUnit(u: DubUnit): void {
    this.units.set(u.id, u);
    for (const e of u.edits) this.edits.push(e);
    this.edits.sort((a, b) => a.at - b.at);
  }

  addAudio(id: number, sampleRate: number, samples: Float32Array): void {
    const buf = this.ctx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
    const wasLive = this.live.has(id);
    this.buffers.set(id, buf);
    if (wasLive) this.stopOne(id); // re-rendered for a new speed; reschedule
  }

  start(): void {
    if (this.active) return;
    this.active = true;
    void this.ctx.resume();
    this.timer = window.setInterval(() => this.tick(), 40);
    this.tick();
  }

  /** Stop audio (user pause, seek, rate change). A freeze edit does NOT call this. */
  halt(): void {
    this.active = false;
    this.freezing = false;
    clearInterval(this.timer);
    clearTimeout(this.freezeTimer);
    for (const id of [...this.live.keys()]) this.stopOne(id);
  }

  seek(): void {
    this.halt();
    this.doneEdits.clear();
  }

  private stopOne(id: number): void {
    const l = this.live.get(id);
    if (l) {
      l.src.onended = null;
      try { l.src.stop(); } catch { /* not started */ }
      this.live.delete(id);
    }
  }

  private nextEditAfter(v: number): Edit | undefined {
    return this.edits.find((e) => e.at >= v - 0.02 && !this.doneEdits.has(`${e.kind}@${e.at}`));
  }

  private tick(): void {
    if (!this.active) return;
    const v = this.clock.now();
    const ctxNow = this.ctx.currentTime;
    const latency = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    const edit = this.nextEditAfter(v);

    if (edit && edit.at - v <= 0.06 / this.speed) this.execute(edit, (edit.at - v) / this.speed);

    for (const [id, u] of this.units) {
      if (this.live.has(id)) continue;
      const buf = this.buffers.get(id);
      if (!buf) continue;
      const d = planStart(u.start, buf.duration, v, ctxNow, this.speed, latency, 1.5, edit && edit.kind === "freeze" ? edit.at : Infinity);
      if (!d) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.gain);
      src.start(Math.max(d.when, ctxNow), d.offset);
      src.onended = () => this.live.delete(id);
      this.live.set(id, { src, planned: u.start });
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

  private execute(e: Edit, inSeconds: number): void {
    this.doneEdits.add(`${e.kind}@${e.at}`);
    const run = () => {
      if (e.kind === "freeze") {
        this.freezing = true;
        this.video.pause();
        this.freezeTimer = window.setTimeout(() => {
          this.video.play();
          this.freezing = false;
        }, (e.added / this.speed) * 1000);
      } else {
        const rates = this.video.rates();
        const want = e.rate * this.speed;
        const r = rates.filter((x) => x >= want - 1e-3).sort((a, b) => a - b)[0] ?? this.speed;
        this.video.setRate(r);
        window.setTimeout(() => this.video.setRate(this.speed), ((e.span / r) * 1000));
      }
    };
    if (inSeconds > 0.004) window.setTimeout(run, inSeconds * 1000);
    else run();
  }

  get scheduledCount(): number {
    return this.live.size;
  }

  dispose(): void {
    this.halt();
    void this.ctx.close();
  }
}
