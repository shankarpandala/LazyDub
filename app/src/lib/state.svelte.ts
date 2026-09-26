import type {
  ClaudeHealth, ClaudeProblem, ClientMessage, CloneStrength, DubUnit, Notice, Range, Speaker, SpeakerScan, VideoInfo,
} from "./types";
import { ASK_AHEAD, END_SLACK, clipLead, contiguousLead, needSeconds, startDecision } from "./buffer";
import { DEFAULTS, normalizeSettings, openMessage, type Settings } from "./settings";
import { addNotice as withNotice, noticeAt } from "./notices";
import { PRIVACY_KEY, healthProblem } from "./claude";

const STAGES = ["idle", "resolving", "fetching", "listening", "translating", "voicing", "ready", "waiting", "error"] as const;
export type Stage = (typeof STAGES)[number];

/** A pipeline stage from the engine's `status`, or null for one this UI doesn't know. */
export function asStage(x: unknown): Stage | null {
  return STAGES.includes(x as Stage) ? (x as Stage) : null;
}

type Send = (m: ClientMessage) => void;

function loadSettings(): Settings {
  try {
    return normalizeSettings(JSON.parse(localStorage.getItem("maata.settings") ?? "{}"));
  } catch {
    return { ...DEFAULTS };
  }
}

function loadFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

export type Recent = { videoId: string; title: string; channel?: string; at: number; position: number };

function loadRecents(): Recent[] {
  try {
    return JSON.parse(localStorage.getItem("maata.recents") ?? "[]");
  } catch {
    return [];
  }
}

export class AppState {
  connection = $state<"connecting" | "open" | "closed">("connecting");
  backend = $state("");
  device = $state("");
  demo = $state(false);
  /** The Claude CLI as the engine found it on connect; null when the engine never calls Claude (the demo). */
  claude = $state<ClaudeHealth | null>(null);
  /** What stops translation now, and when the engine said so (ms): the banner's countdown runs from then. */
  claudeProblem = $state<(ClaudeProblem & { at: number }) | null>(null);
  /** The one-time notice that transcript text goes to Anthropic has been read. */
  privacySeen = $state(loadFlag(PRIVACY_KEY));

  video = $state<VideoInfo | null>(null);
  stage = $state<Stage>("idle");
  message = $state("");
  error = $state<{ message: string; retryable: boolean } | null>(null);

  units = $state<DubUnit[]>([]);
  /** Distinct lines dubbed so far. */
  lineCount = $state(0);
  skipped = $state<{ id: number; start: number; end: number }[]>([]);
  readyUntil = $state(0);
  /** Merged dubbed ranges from the engine (video seconds). */
  readyRanges = $state<Range[]>([]);
  /** Engine speed, × realtime (0 = not measured yet). */
  throughput = $state(0);
  /** Lead the engine recommends before playing (s). */
  targetLead = $state(0);
  /** Spans whose audio the UI doesn't hold yet: evicted, or held back by the engine beyond its window. */
  missing = $state<Range[]>([]);
  /** "Prepare the whole video" is on for this video. */
  prepareAll = $state(false);
  speakers = $state<Speaker[]>([]);
  speakerScan = $state<SpeakerScan | null>(null);
  /** Stretches the engine flagged for the listener (merged, by video time). */
  notices = $state<Notice[]>([]);
  /** The engine's look-ahead for this video (the setting when it was opened). */
  lookahead = $state(DEFAULTS.lookahead);
  /** Voice match this video was opened with (the setting when it was opened). */
  cloneStrength = $state<CloneStrength>(DEFAULTS.cloneStrength);

  playing = $state(false);
  time = $state(0);
  duration = $state(0);
  speed = $state(1);
  volume = $state(1);
  waitingForDub = $state(false);
  /** The video has played at least once. */
  started = $state(false);
  /** Playback ran out of dub; the next start waits for the full `need`. */
  ranDry = $state(false);

  settings = $state<Settings>(loadSettings());
  settingsOpen = $state(false);
  recents = $state<Recent[]>(loadRecents());
  windowSeconds = $state<number[]>([]);

  remaining = $derived(this.duration > 0 ? Math.max(0, this.duration - this.time) : Infinity);
  /** Contiguous Telugu ready ahead of the playhead (s); audio not held yet stops it only near the playhead. */
  lead = $derived(clipLead(this.time, contiguousLead(this.time, this.readyRanges, this.readyUntil), this.missing, ASK_AHEAD));
  /** Telugu to bank before (re)starting (s). */
  need = $derived(needSeconds({
    remaining: this.remaining, prepareAhead: this.settings.prepareAhead, targetLead: this.targetLead, lookahead: this.lookahead,
  }));
  /** Everything from the playhead to the end is dubbed. */
  allReady = $derived(this.duration > 0 && this.lead >= this.remaining - END_SLACK);
  /** Pressing play now would wait for the dub. */
  mustWait = $derived(startDecision(this.lead, this.need, !this.started || this.ranDry) === "wait");
  current = $derived(this.units.find((u) => this.time >= u.start && this.time <= u.start + u.audioWall + 0.3) ?? null);
  /** The notice at the playhead, if any. */
  notice = $derived(noticeAt(this.notices, this.time));

  /**
   * Start dubbing a video: forget the last one, then ask the engine to open this one with the current settings. What
   * the settings were at this moment is what the video is dubbed with (lookahead, voice match) until the next open.
   */
  openVideo(url: string, send: Send): void {
    this.resetVideo();
    this.stage = "resolving";
    this.message = "Finding the video…";
    send(openMessage(url, this.settings));
  }

  /** Switch a speaker to or from the stock voice. Shown at once; the engine's next `speakers` message is the truth. */
  setPreset(speaker: string, usePreset: boolean, send: Send): void {
    this.speakers = this.speakers.map((s) => (s.id === speaker ? { ...s, usePreset } : s));
    send({ type: "speaker_preset", speaker, usePreset });
  }

  /** The engine's `status`. A stage this UI doesn't know is ignored, message and all, instead of breaking the pipeline view. */
  setStatus(stage: string, message: string): void {
    const s = asStage(stage);
    if (!s) return;
    this.stage = s;
    this.message = message;
  }

  addNotice(n: Notice): void {
    this.notices = withNotice(this.notices, n);
  }

  /** The engine's `hello`: how the Claude CLI is, and anything wrong with it already. */
  setClaude(health: ClaudeHealth | null | undefined, now: number = Date.now()): void {
    this.claude = health ?? null;
    const p = healthProblem(health);
    this.claudeProblem = p ? { ...p, at: now } : null;
  }

  claudeError(p: ClaudeProblem, now: number = Date.now()): void {
    this.claudeProblem = { ...p, at: now };
  }

  claudeOk(): void {
    this.claudeProblem = null;
  }

  ackPrivacy(): void {
    this.privacySeen = true;
    try {
      localStorage.setItem(PRIVACY_KEY, "1");
    } catch {
      // storage off: the notice shows again next time
    }
  }

  saveSettings(): void {
    localStorage.setItem("maata.settings", JSON.stringify(this.settings));
  }

  remember(): void {
    if (!this.video) return;
    const r: Recent = { videoId: this.video.videoId, title: this.video.title || "Untitled video", channel: this.video.channel, at: Date.now(), position: this.time };
    this.recents = [r, ...this.recents.filter((x) => x.videoId !== r.videoId)].slice(0, 8);
    localStorage.setItem("maata.recents", JSON.stringify(this.recents));
  }

  resetVideo(): void {
    this.video = null;
    this.units = [];
    this.lineCount = 0;
    this.skipped = [];
    this.readyUntil = 0;
    this.readyRanges = [];
    this.throughput = 0;
    this.targetLead = 0;
    this.missing = [];
    this.prepareAll = false;
    this.speakers = [];
    this.speakerScan = null;
    this.notices = [];
    this.lookahead = this.settings.lookahead;
    this.cloneStrength = this.settings.cloneStrength;
    this.playing = false;
    this.time = 0;
    this.duration = 0;
    this.error = null;
    this.stage = "idle";
    this.message = "";
    this.waitingForDub = false;
    this.started = false;
    this.ranDry = false;
  }
}

export const app = new AppState();
