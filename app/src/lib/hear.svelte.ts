import type { AudioFrame } from "./engine";

/** How long a Hear voice request waits for its frame before the button gives up (ms). */
const WAIT_MS = 6000;

/**
 * The speaker check's Hear voice (OFFLINE-RENDER §5): the engine sends a speaker's sample as one binary frame (ADR-007),
 * played here through Web Audio, one at a time. The only audio the app plays.
 */
class VoicePlayer {
  /** The speaker whose sample was asked for and hasn't come yet. */
  waiting = $state<string | null>(null);
  /** The speaker whose sample is playing. */
  playing = $state<string | null>(null);
  private ctx: AudioContext | null = null;
  private source: AudioBufferSourceNode | null = null;
  private timer = 0;

  /** Inside the click that asks for a sample: WebKit starts audio output only from a user gesture. */
  ask(speaker: string): void {
    this.stop();
    this.ctx ??= new AudioContext();
    void this.ctx.resume();
    this.waiting = speaker;
    clearTimeout(this.timer);
    this.timer = window.setTimeout(() => (this.waiting = null), WAIT_MS); // a stock voice has no sample: nothing comes
  }

  /** A sample frame arrived for `speaker`. */
  play(speaker: string, frame: AudioFrame): void {
    if (this.waiting !== speaker) return; // asked for another since
    clearTimeout(this.timer);
    this.waiting = null;
    const ctx = (this.ctx ??= new AudioContext());
    const buffer = ctx.createBuffer(1, Math.max(1, frame.samples.length), frame.sampleRate);
    buffer.getChannelData(0).set(frame.samples);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    src.onended = () => {
      if (this.source === src) {
        this.source = null;
        this.playing = null;
      }
    };
    this.source = src;
    this.playing = speaker;
    src.start();
  }

  stop(): void {
    const src = this.source;
    this.source = null;
    this.playing = null;
    try {
      src?.stop();
    } catch {
      // already ended
    }
  }
}

export const hear = new VoicePlayer();
