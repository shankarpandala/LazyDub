/**
 * Buffer policy (replaces the spec §6.7 8/2/12 s hysteresis): bank a long, contiguous stretch of
 * Telugu before playing, then play until the dub truly runs out. No small-lead stop-and-go.
 *
 * - Before the first play, and after running dry, wait until `lead ≥ need`, where
 *   `need = min(remaining, max(min(prepareAhead, 0.8 × lookahead), targetLead))`. targetLead comes from the engine: the
 *   lead that plays to the end without a stall at its throughput r, (1 − r) × remaining + a margin. The engine works
 *   past its look-ahead to bank it while the video waits, so it isn't capped; the wait is (need − lead) ÷ r of wall time
 *   (ARCHITECTURE §3.12).
 * - While playing, pause only when `lead < RUN_DRY` and no line is sounding.
 * - After a user pause or a seek, carry on if at least `min(need, RESUME_MIN)` is ready.
 *
 * `lead` is the contiguous dubbed time ahead of the playhead, from the engine's merged ranges.
 */
import { mergeRanges } from "./format";
import type { Span } from "./sync";
import type { Range } from "./types";

/** Below this lead (s) with nothing sounding, the dub has run out. */
export const RUN_DRY = 0.5;
/** "Play now" needs at least this much ready (s), or it would stop again at once. */
export const PLAY_NOW_MIN = 2;
/** A resume after a user pause or a seek carries on with at least this much ready (s, capped by need). */
export const RESUME_MIN = 30;
/**
 * YouTube's duration and the engine's audio can disagree by a second or so: the last END_SLACK
 * seconds of the video needn't be dubbed before playing through to the end.
 */
export const END_SLACK = 2;
/** Ranges this close (s) count as one: float rounding between the engine's chunks. */
const TOUCH = 0.25;
/**
 * Share of the engine's look-ahead the user's prepare-ahead setting may ask for: an engine fast enough for playback dubs
 * no further than playhead + look-ahead, and its ready ranges grow a whole ASR chunk at a time.
 */
const LOOKAHEAD_USE = 0.8;
/**
 * Audio the UI doesn't hold (evicted, or held back by the engine outside its window) is asked for once it is this close
 * ahead of the playhead (s). Only such near audio stops the lead: the engine keeps the rest on disk and sends it when asked.
 */
export const ASK_AHEAD = 90;

/** Contiguous dubbed seconds ahead of `time`. `until` is only a fallback for an engine that sends no ranges. */
export function contiguousLead(time: number, ranges: Range[], until: number): number {
  if (!ranges.length) return Math.max(0, until - time);
  for (const [a, b] of mergeRanges(ranges, TOUCH)) if (a - TOUCH <= time && time < b) return b - time;
  return 0;
}

/**
 * Cut the lead at the first span whose audio the UI doesn't hold and is waiting to get (evicted, or held back by the
 * engine), if it starts within `within` seconds of `time`: further ones are on the engine's disk, asked for in time.
 */
export function clipLead(time: number, lead: number, missing: Range[], within = Infinity): number {
  let out = lead;
  for (const [a, b] of missing) if (b > time && a < time + Math.min(out, within)) out = Math.max(0, a - time);
  return out;
}

export type NeedInputs = {
  /** Video seconds left to the end (Infinity if unknown). */
  remaining: number;
  /** The user's "prepare before playing" setting (s). */
  prepareAhead: number;
  /** The engine's recommendation (s; 0 = unknown). */
  targetLead: number;
  /** The engine's look-ahead (s): neither the setting nor the engine's target can ask for more than it will bank. */
  lookahead?: number;
};

/** Seconds of contiguous Telugu to bank before starting. */
export function needSeconds({ remaining, prepareAhead, targetLead, lookahead }: NeedInputs): number {
  // The setting is capped by what the look-ahead banks. The engine's target (pacing.target_lead) isn't: a slower engine
  // keeps working past its look-ahead to bank it (session._horizon), and a capped target ran dry partway through a long
  // video (research gap-1: 480 s banked at 0.59x lasts about 19.5 min).
  const setting = lookahead ? Math.min(prepareAhead, lookahead * LOOKAHEAD_USE) : prepareAhead;
  return Math.max(0, Math.min(remaining - END_SLACK, Math.max(setting, targetLead || 0)));
}

/** Fraction of `need` that is ready, 0–1. */
export function progress(lead: number, need: number): number {
  return need <= 0 ? 1 : Math.min(1, Math.max(0, lead / need));
}

/**
 * Wall seconds until `need` is banked while the video waits: the playhead is still, so the lead grows at the engine's
 * throughput r, and the wait is the missing lead ÷ r (ARCHITECTURE §3.12). null when throughput is unknown.
 */
export function etaSeconds(lead: number, need: number, throughput: number): number | null {
  if (!(throughput > 0)) return null;
  if (lead >= need) return 0;
  return (need - lead) / Math.max(throughput, 0.01);
}

export function fmtEta(s: number): string {
  if (s < 10) return "a few seconds left";
  if (s < 60) return `about ${Math.ceil(s / 10) * 10} s left`;
  if (s < 3600) return `about ${Math.round(s / 60)} min left`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return `about ${h} h${m ? ` ${m} min` : ""} left`;
}

/** Whether pressing play (or a seek while playing) may go straight on, or must wait for the dub. */
export function startDecision(lead: number, need: number, mustFill: boolean): "play" | "wait" {
  const want = mustFill ? need : Math.min(need, RESUME_MIN);
  return lead >= want - 1e-6 ? "play" : "wait";
}

export type BufferSnapshot = {
  /** The user wants playback (pressed play and hasn't paused). */
  wantPlay: boolean;
  /** Paused by us, waiting for the dub. */
  waiting: boolean;
  /** The video clock is running. */
  playing: boolean;
  /** A planned freeze-frame holds the video. */
  freezing: boolean;
  /** A dub line is sounding now. */
  linePlaying: boolean;
  lead: number;
  need: number;
  remaining: number;
};

/** One tick of the policy: start the video, stall it, or leave it be. */
export function bufferStep(s: BufferSnapshot): "start" | "stall" | null {
  if (!s.wantPlay) return null;
  if (s.waiting) return s.lead >= s.need - 1e-6 ? "start" : null;
  // Not dry if all that's missing is the end slack; this also keeps a start from stalling again at once.
  if (s.playing && !s.freezing && !s.linePlaying && s.lead < RUN_DRY && s.remaining - s.lead > END_SLACK) return "stall";
  return null;
}

/** After a seek to t, the engine re-sends the dubbed lines starting in [t - RESEND_BEHIND, t + look-ahead). */
export const RESEND_BEHIND = 5;
/** Missing audio the playhead nears is asked for (an `audio` message) at most this often (ms)... */
export const REASK_MS = 20_000;
/** ...and no longer waited on after this many asks. */
export const MAX_ASKS = 2;

type Gone = { start: number; end: number; asked: number; asks: number };

/**
 * Lines whose audio the UI doesn't hold, until the engine sends it: evicted (§6.8), or held back by the engine outside
 * the window near the playhead (ARCHITECTURE §3.12). `lead` stops at those near the playhead, so a return to evicted
 * video waits for its audio. They are asked for by id as the playhead nears them (`ask`); a seek re-sends lines from
 * RESEND_BEHIND before the seek point on, so a line that began earlier than that is no longer waited on after one.
 */
export class MissingAudio {
  private gone = new Map<number, Gone>();
  /** The spans changed since `ranges()` was last read. */
  dirty = false;

  get size(): number {
    return this.gone.size;
  }

  clear(): void {
    this.gone.clear();
    this.dirty = false;
  }

  add(spans: Span[]): void {
    for (const s of spans) this.gone.set(s.id, { start: s.start, end: s.end, asked: 0, asks: 0 });
    if (spans.length) this.dirty = true;
  }

  /** The audio for `id` arrived. */
  got(id: number): void {
    if (this.gone.delete(id)) this.dirty = true;
  }

  /** Stop waiting on lines a seek to `t` can't bring back: they began more than RESEND_BEHIND before it. */
  forgetBefore(t: number): void {
    for (const [id, g] of this.gone) {
      if (g.end > t && g.start < t - RESEND_BEHIND) {
        this.gone.delete(id);
        this.dirty = true;
      }
    }
  }

  /** A seek to `t` went to the engine at `ms`: count it as an ask for the lines it re-sends. */
  asked(t: number, lookahead: number, ms: number): void {
    this.forgetBefore(t);
    for (const g of this.gone.values()) {
      if (g.end > t && g.start < t + lookahead) {
        g.asked = ms;
        g.asks++;
      }
    }
  }

  /**
   * The lines between `time` and `horizon` due an ask, counted as asked at `ms`: never asked, or last asked REASK_MS
   * ago. Lines already asked MAX_ASKS times are no longer waited on.
   */
  ask(time: number, horizon: number, ms: number): number[] {
    const ids: number[] = [];
    for (const [id, g] of this.gone) {
      if (g.end <= time || g.start >= horizon || (g.asked && ms - g.asked < REASK_MS)) continue;
      if (g.asks >= MAX_ASKS) {
        this.gone.delete(id);
        this.dirty = true;
      } else {
        g.asked = ms;
        g.asks++;
        ids.push(id);
      }
    }
    return ids;
  }

  /** Merged missing spans, for `clipLead`. */
  ranges(): Range[] {
    this.dirty = false;
    return mergeRanges([...this.gone.values()].map((g): Range => [g.start, g.end]), 0);
  }
}
