import type { ClientMessage, CloneStrength, TranslationStyle, TtsScript } from "./types";

/** Speed-up range for a Telugu line. Above about 1.2× speech starts to sound rushed. */
export const SPEED_CAP_MIN = 1;
export const SPEED_CAP_MAX = 1.25;
export const SPEED_CAP_STEP = 0.05;

/** Seconds of Telugu to bank before playing. */
export const PREPARE_OPTIONS = [60, 120, 300, 600] as const;
/** How far ahead of the playhead the engine dubs (s). */
export const LOOKAHEAD_OPTIONS = [300, 600, 1200] as const;

/**
 * Translation styles, named by register, not by how much English they use: the spoken style sets no amount of English
 * (spec §5). Many lines have none; English stays only where Telugu speakers really say it.
 */
export const STYLE_OPTIONS: readonly { id: TranslationStyle; label: string; hint: string }[] = [
  {
    id: "colloquial", label: "Everyday spoken Telugu",
    hint: "How people really talk. Names and modern terms stay in English only where Telugu speakers would say them that way.",
  },
  { id: "formal", label: "More formal Telugu", hint: "Closer to written Telugu." },
];

/**
 * How the Telugu voice reads English words (decision D6). Telugu script is what the voice was trained on; Latin letters
 * are there for the maintainer's blind listening A/B and may be retired after it.
 */
export const TTS_SCRIPT_OPTIONS: readonly { id: TtsScript; label: string; hint: string }[] = [
  { id: "telugu", label: "As Telugu speakers write them", hint: "English words spelled in Telugu script: ఫోన్, సెట్టింగ్స్." },
  { id: "latin", label: "In English letters (A/B test)", hint: "The same line with its English words in Latin letters, to compare by ear." },
];

const STYLES: readonly TranslationStyle[] = STYLE_OPTIONS.map((o) => o.id);
const TTS_SCRIPTS: readonly TtsScript[] = TTS_SCRIPT_OPTIONS.map((o) => o.id);
const CLONE_STRENGTHS: readonly CloneStrength[] = ["closest", "balanced", "natural"];
const THEMES = ["dark", "light", "system"] as const;

export type Theme = (typeof THEMES)[number];

export type Settings = {
  /** Most a Telugu line may be sped up to fit its slot (SPEED_CAP_MIN–SPEED_CAP_MAX). */
  speedCap: number;
  /** Rarely hold the picture for a moment when a line still can't fit. The video is never slowed down. */
  allowFreeze: boolean;
  style: TranslationStyle;
  /**
   * How closely cloned voices follow each speaker, against how natively they speak Telugu. One value for every
   * speaker in the video; the engine reports the match each voice was actually built with.
   */
  cloneStrength: CloneStrength;
  /** How the Telugu voice reads English words (TTS_SCRIPT_OPTIONS). */
  ttsScript: TtsScript;
  /** Seconds of Telugu to bank before playing (PREPARE_OPTIONS). */
  prepareAhead: number;
  /** How far ahead of the playhead the engine dubs (LOOKAHEAD_OPTIONS). */
  lookahead: number;
  updateCheck: boolean;
  hud: boolean;
  theme: Theme;
};

/**
 * Voice match when none is chosen. It is the engine's own default (the session falls back to it when `open` has no
 * cloneStrength), so both sides agree. It stays "closest" until experiment E3 settles the default by ear (spec §4).
 */
export const DEFAULT_CLONE_STRENGTH: CloneStrength = "closest";

export const DEFAULTS: Readonly<Settings> = {
  speedCap: 1.2, allowFreeze: true, style: "colloquial", cloneStrength: DEFAULT_CLONE_STRENGTH, ttsScript: "telugu",
  prepareAhead: 300, lookahead: 600, updateCheck: true, hud: false, theme: "dark",
};

/** Clamp to the allowed speed-up range, on its 0.05 steps; anything unreadable is the default. */
export function clampSpeedCap(x: unknown): number {
  if (typeof x !== "number" || !Number.isFinite(x)) return DEFAULTS.speedCap;
  const perUnit = Math.round(1 / SPEED_CAP_STEP);
  const stepped = Math.round(x * perUnit) / perUnit;
  return Math.min(SPEED_CAP_MAX, Math.max(SPEED_CAP_MIN, stepped));
}

const oneOf = <T>(options: readonly T[], x: unknown, fallback: T): T => (options.includes(x as T) ? (x as T) : fallback);
const bool = (x: unknown, fallback: boolean): boolean => (typeof x === "boolean" ? x : fallback);

/**
 * Settings from storage, made valid. Settings saved by an older build carry keys that are gone
 * (allowSlowdown, maxPause, voiceMode) or a speed-up above today's cap: those are dropped or clamped.
 */
export function normalizeSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  return {
    speedCap: clampSpeedCap(r.speedCap),
    allowFreeze: bool(r.allowFreeze, DEFAULTS.allowFreeze),
    style: oneOf(STYLES, r.style, DEFAULTS.style),
    cloneStrength: oneOf(CLONE_STRENGTHS, r.cloneStrength, DEFAULTS.cloneStrength),
    ttsScript: oneOf(TTS_SCRIPTS, r.ttsScript, DEFAULTS.ttsScript),
    prepareAhead: oneOf<number>(PREPARE_OPTIONS, r.prepareAhead, DEFAULTS.prepareAhead),
    lookahead: oneOf<number>(LOOKAHEAD_OPTIONS, r.lookahead, DEFAULTS.lookahead),
    updateCheck: bool(r.updateCheck, DEFAULTS.updateCheck),
    hud: bool(r.hud, DEFAULTS.hud),
    theme: oneOf(THEMES, r.theme, DEFAULTS.theme),
  };
}

/** The `open` message for a video: the settings the engine dubs it with, fixed until the next open. */
export function openMessage(url: string, s: Settings): Extract<ClientMessage, { type: "open" }> {
  return {
    type: "open", url, style: s.style, lookahead: s.lookahead,
    cloneStrength: s.cloneStrength, speedCap: clampSpeedCap(s.speedCap), allowFreeze: s.allowFreeze, ttsScript: s.ttsScript,
  };
}
