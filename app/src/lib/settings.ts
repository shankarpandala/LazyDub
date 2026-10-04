import type { DubOptions, JobSettings, TranslationStyle, TtsScript } from "./types";

/** Speed-up range for a Telugu line. Above about 1.2× speech starts to sound rushed. */
export const SPEED_CAP_MIN = 1;
export const SPEED_CAP_MAX = 1.25;
export const SPEED_CAP_STEP = 0.05;

/** New dub's "First 15 minutes" (OFFLINE-RENDER §3). */
export const PREVIEW_AT = 900;

/** Speakers: Auto, or a count the user knows (the engine's auto range). */
export const SPEAKER_OPTIONS: readonly ("auto" | number)[] = ["auto", 1, 2, 3, 4, 5, 6];

/**
 * Translation styles, named by register, not by how much English they use: the everyday style sets no amount of
 * English (spec §5). English stays where ordinary Telugu people say it in English.
 */
export const STYLE_OPTIONS: readonly { id: TranslationStyle; label: string; hint: string }[] = [
  {
    id: "colloquial", label: "Everyday spoken",
    hint: "How ordinary Telugu people talk, with English words only where they'd say them that way.",
  },
  { id: "formal", label: "More formal", hint: "Closer to written Telugu." },
];

/**
 * How the Telugu voice reads English words (decision D6). Telugu script is what the voice was trained on; Latin letters
 * are there for the maintainer's blind listening A/B and may be retired after it.
 */
export const TTS_SCRIPT_OPTIONS: readonly { id: TtsScript; label: string; hint: string }[] = [
  { id: "telugu", label: "As Telugu speakers write them", hint: "English words spelled in Telugu script: ఫోన్, సెట్టింగ్స్." },
  { id: "latin", label: "In English letters (A/B test)", hint: "The same line with its English words in Latin letters, to compare by ear." },
];

export const DEFAULT_OPTIONS: Readonly<DubOptions> = {
  speakers: "auto", style: "colloquial", stopAt: null, speedCap: 1.2, ttsScript: "telugu",
};

/** Clamp to the allowed speed-up range, on its 0.05 steps; anything unreadable is the default. */
export function clampSpeedCap(x: unknown): number {
  if (typeof x !== "number" || !Number.isFinite(x)) return DEFAULT_OPTIONS.speedCap;
  const perUnit = Math.round(1 / SPEED_CAP_STEP);
  const stepped = Math.round(x * perUnit) / perUnit;
  return Math.min(SPEED_CAP_MAX, Math.max(SPEED_CAP_MIN, stepped));
}

const oneOf = <T>(options: readonly T[], x: unknown, fallback: T): T => (options.includes(x as T) ? (x as T) : fallback);

/**
 * A job's options, made valid: what `prepare` sends to continue it (the whole video, or the same again). Its
 * speakers stay its own (they belong to its video).
 */
export function jobOptions(s: Partial<JobSettings> | null | undefined): DubOptions {
  const r = s ?? {};
  const k = r.speakers;
  return {
    speakers: typeof k === "number" && SPEAKER_OPTIONS.includes(k) ? k : "auto",
    style: oneOf(STYLE_OPTIONS.map((o) => o.id), r.style, DEFAULT_OPTIONS.style),
    stopAt: r.stopAt === PREVIEW_AT ? PREVIEW_AT : null,
    speedCap: clampSpeedCap(r.speedCap),
    ttsScript: oneOf(TTS_SCRIPT_OPTIONS.map((o) => o.id), r.ttsScript, DEFAULT_OPTIONS.ttsScript),
  };
}

/**
 * New dub's options, starting from the last job's (§5): its style, length, speed-up cap and TTS script. Not its
 * speaker count, which was that video's: a new video starts at Auto.
 */
export function newDubOptions(last: Partial<JobSettings> | null | undefined): DubOptions {
  return { ...jobOptions(last), speakers: "auto" };
}
