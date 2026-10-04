/**
 * The engine protocol of the background dub (OFFLINE-RENDER §4): the engine owns the jobs, one runs at a time and the
 * others wait in its queue; the UI shows them, sets them up and saves nothing of its own.
 */

export type TranslationStyle = "colloquial" | "formal";

/**
 * What the Telugu voice reads (decision D6): English words in Telugu script (the default), or in Latin letters, for
 * comparing the two by ear. The engine rebuilds the Latin form from the same translation.
 */
export type TtsScript = "telugu" | "latin";

/** The engine's stages, in order (`render.STAGES`). */
export type StageKey =
  | "fetch" | "speakers" | "transcript" | "units" | "voices" | "brief" | "separate" | "translate" | "voice_lines"
  | "finish" | "video" | "export";

export type StageState = "todo" | "running" | "done" | "failed";

/** One stage of a `render`: its progress in its own unit, the seconds it took and what it has left (null: not running). */
export type StageRow = {
  key: StageKey;
  label: string;
  state: StageState;
  done: number;
  total: number;
  unit: string;
  seconds: number;
  eta: number | null;
  /** What the stage is doing besides its count ("Waiting for the background sound", "Finishing the file…"). */
  extra?: string;
};

export type JobStatus = "queued" | "running" | "paused" | "interrupted" | "waiting" | "failed" | "done";

/**
 * The saved MP4 (§2.17). `missing`: the file isn't at its path any more (moved or deleted). `bed`: the Telugu voices are
 * over the original music and sounds (false: the voices only, `warning` says why).
 */
export type Output = {
  path: string;
  bytes: number;
  kind: "whole" | "preview";
  at: number;
  loudness: { I: number; TP: number; LRA: number } | null;
  warning: string | null;
  bed?: boolean;
  missing: boolean;
  note?: string;
};

/** The Mac slept while the job ran: the last progress before (`at`) and the first after (`resumed`), epoch seconds. */
export type Slept = { at: number; resumed: number };

/** Coverage classes of the wordings voiced (C complete, m minor, P partial, E error), and the lines outside them. */
export type Coverage = {
  C: number; m: number; P: number; E: number;
  otherTier: number; unreviewed: number; skipped: number; lines: number;
};

/** The final lines (job.json `report`): those flagged long or unreviewed, and the overdraft seconds (§2.11). */
export type Report = { lines: number; long: number; unreviewed: number; overdraft: number; carried: number };

/** A job's progress, to every window, at most twice a second and on every state change. */
export type Render = {
  videoId: string;
  status: JobStatus;
  stage: StageKey | null;
  stages: StageRow[];
  stopAt: number | null;
  finalUntil: number | null;
  eta: number | null;
  elapsed: number;
  /** Its place in the queue (1: next), or null. */
  position: number | null;
  output: Output | null;
  slept: Slept | null;
  coverage: Coverage | null;
  report?: Report | null;
  error: string | null;
};

/** The options of a job (job.json `settings`). */
export type JobSettings = {
  speakers: number | null;
  style: TranslationStyle;
  stopAt: number | null;
  speedCap: number;
  ttsScript: TtsScript;
  /** Speakers voiced with a stock voice instead of their clone. */
  presets: string[];
};

/** A job of the library (`renders`), newest activity first. */
export type JobItem = {
  videoId: string;
  title: string;
  channel: string;
  duration: number;
  /** The cached thumbnail's loopback URL, or null (none yet, or the demo's). */
  thumb: string | null;
  status: JobStatus;
  position: number | null;
  stopAt: number | null;
  settings: Partial<JobSettings> | null;
  finalUntil: number | null;
  stage: StageKey | null;
  eta: number | null;
  updatedAt: number | null;
  createdAt: number | null;
  bytes: number;
  output: Output | null;
  slept: Slept | null;
  error: string | null;
  expired: boolean;
  /** What the job view shows while the job sends no `render` (it isn't running): its stages (no ETA) and report. */
  stages: StageRow[];
  elapsed: number;
  coverage: Coverage | null;
  report: Report | null;
};

/** Low and high, from the engine's priors. */
export type Span = [number, number];

export type Estimate = {
  seconds: Span;
  claudeCalls: Span;
  /** The first 15 minutes'. */
  preview?: { seconds: Span; claudeCalls: Span };
  /** Seconds of work before a new job would start (null: nothing runs). */
  ahead: number | null;
};

/** `inspect`'s answer: a video and its job, if it has one. */
export type VideoInfo = {
  videoId: string;
  start: number;
  title: string;
  channel: string;
  duration: number;
  thumbnail: string;
  render: JobItem | null;
  estimate: Estimate;
};

export type FoundSpeaker = {
  id: string;
  label: string;
  talkSeconds: number;
  share: number;
  firstAt: number;
  turns: number;
  /** Their share of each of 120 equal slices of the video. */
  activity: number[];
};

/** The speaker check (§2.3): who the whole-file diarization found, and what it merged. */
export type SpeakersFound = {
  videoId: string;
  mode: "auto" | "hint";
  fresh: boolean;
  /** Seconds, by the priors when it was sent, until the Telugu speech starts: correcting is free until then. */
  freeFor: number;
  bounds: [number, number] | null;
  speakers: FoundSpeaker[];
  merged: { from: string; into: string; why: string; talkSeconds: number }[];
};

export type EngineSettings = { outputDir: string };

/** Why translation through the Claude CLI can't go on, as the engine classes it (ADR-019). */
export type ClaudeProblemKind =
  | "missing" | "not_signed_in" | "outdated" | "usage_limit" | "transient" | "stalled" | "timeout" | "bad_output" | "failed";

/** The Claude CLI as the engine found it when the UI connected. */
export type ClaudeHealth = {
  installed: boolean;
  version: string | null;
  /** null: the CLI couldn't say. */
  signedIn: boolean | null;
  /** Models this CLI version can run. */
  models: string[];
  /** The model Maata asks for. */
  model: string;
  problem: ClaudeProblemKind | null;
  message: string;
};

/**
 * A Claude call failed in a way translation can't get past on its own. Translation waits `retryIn` s (for a usage
 * limit, until it resets), then tries again; the work on this Mac goes on meanwhile.
 */
export type ClaudeProblem = {
  kind: ClaudeProblemKind;
  message: string;
  /** Which usage limit: session, weekly, opus, sonnet or overage, when known. */
  limit?: string | null;
  /** Epoch seconds the usage limit resets, when known. */
  resetsAt?: number | null;
  retryIn?: number | null;
  /** The job it held back. */
  videoId?: string;
};

export type EngineMessage =
  /** claude: null when the engine never calls Claude (the demo engine). */
  | {
      type: "hello"; backend: string; device: string; demo: boolean; claude?: ClaudeHealth | null;
      renders: JobItem[]; render: Render | null; settings: EngineSettings;
    }
  /** `url`: the link `inspect` asked about, as sent. */
  | ({ type: "video"; url?: string } & VideoInfo)
  | ({ type: "render" } & Render)
  | { type: "renders"; items: JobItem[] }
  | ({ type: "speakers_found" } & SpeakersFound)
  | ({ type: "settings" } & EngineSettings)
  | ({ type: "claude_error" } & ClaudeProblem)
  /** A Claude call went through again after a claude_error. */
  | { type: "claude_ok"; videoId?: string }
  /** `url`: an `inspect` that failed, the link it asked about. */
  | { type: "error"; message: string; retryable: boolean; url?: string };

/** New dub's options, as `prepare` sends them. */
export type DubOptions = {
  speakers: "auto" | number;
  style: TranslationStyle;
  stopAt: number | null;
  speedCap: number;
  ttsScript: TtsScript;
};

export type ClientMessage =
  | { type: "inspect"; url: string }
  | ({ type: "prepare"; url: string } & DubOptions)
  | { type: "pause"; videoId: string }
  | { type: "resume"; videoId: string }
  /** Refused while the job runs: the UI pauses it first. `forget` deletes its translations too. */
  | { type: "remove"; videoId: string; forget?: boolean }
  | { type: "set_speakers"; videoId: string; speakers: "auto" | number }
  | { type: "set_voice"; videoId: string; speaker: string; usePreset: boolean }
  | { type: "renders" }
  /** One binary frame back: the speaker's Hear voice sample (engine.ts SAMPLE_ID). */
  | { type: "voice_sample"; videoId: string; speaker: string }
  /** The engine opens the job's own MP4 (or shows it in Finder); never a path the UI sends. */
  | { type: "open_output"; videoId: string; reveal: boolean }
  | { type: "settings"; outputDir: string };
