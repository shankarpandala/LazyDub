/**
 * VideoClock (ADR-008): a smooth, extrapolated estimate of the player's video time.
 *
 * YouTube's IFrame API only reports `getCurrentTime()` coarsely over postMessage, so the clock
 * extrapolates from an anchor using `performance.now()` and the current rate, re-anchoring
 * hard on large errors and slewing gently on small ones.
 */
export class VideoClock {
  private anchorVideo = 0;
  private anchorMs = 0;
  rate = 1;
  playing = false;

  /** Errors above this (s) re-anchor immediately; below it the clock slews. */
  hardResync = 0.25;
  slew = 0.08;

  now(ms: number = performance.now()): number {
    if (!this.playing) return this.anchorVideo;
    return this.anchorVideo + ((ms - this.anchorMs) / 1000) * this.rate;
  }

  anchor(video: number, ms: number = performance.now()): void {
    this.anchorVideo = video;
    this.anchorMs = ms;
  }

  setPlaying(playing: boolean, video?: number, ms: number = performance.now()): void {
    const v = video ?? this.now(ms);
    this.playing = playing;
    this.anchor(v, ms);
  }

  setRate(rate: number, ms: number = performance.now()): void {
    const v = this.now(ms);
    this.rate = rate;
    this.anchor(v, ms);
  }

  /** Feed a reported player time; returns the error before correction (s). */
  observe(reported: number, ms: number = performance.now()): number {
    const err = reported - this.now(ms);
    if (!this.playing || Math.abs(err) > this.hardResync) {
      this.anchor(reported, ms);
    } else {
      this.anchor(this.now(ms) + err * this.slew, ms);
    }
    return err;
  }
}
