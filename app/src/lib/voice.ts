import type { CloneStrength, DubUnit, Speaker, SpeakerStatus } from "./types";
import { fmtTime } from "./format";

/**
 * Voice match options, strongest likeness first. The timbre always comes from the speaker. None is marked as the
 * recommended one: the maintainer picks the default by ear (spec §4, experiment E3).
 */
export const CLONE_STRENGTH_OPTIONS: readonly { id: CloneStrength; label: string; hint: string }[] = [
  { id: "closest", label: "Closest to original", hint: "Keeps more of the speaker's own delivery and accent. Can sound less like natural Telugu." },
  { id: "balanced", label: "Balanced", hint: "The speaker's voice with a more natural Telugu delivery." },
  { id: "natural", label: "Most natural Telugu", hint: "Fluent, native-sounding Telugu. Sounds a little less like the original speaker." },
];

export function cloneStrengthLabel(c: CloneStrength): string {
  return CLONE_STRENGTH_OPTIONS.find((o) => o.id === c)?.label ?? c;
}

/** A speaker's voice as the panel shows it: the listener's preset choice wins over the engine's status. */
export function speakerStatus(s: Speaker): SpeakerStatus {
  if (s.usePreset) return "preset";
  return s.status ?? (s.voice === "cloned" ? "cloned" : "found");
}

/**
 * Why a speaker is on the stock voice: the listener chose it ("chosen"), or the engine has no clone for
 * them ("fallback"). null while they are, or are about to be, cloned.
 */
export function presetReason(s: Speaker): "chosen" | "fallback" | null {
  if (s.usePreset) return "chosen";
  return speakerStatus(s) === "preset" ? "fallback" : null;
}

/** Lines dubbed in the stock voice, per speaker id: what a silent fallback would otherwise hide. */
export function presetLineCounts(units: readonly DubUnit[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const u of units) if (u.voice === "preset") out.set(u.speaker, (out.get(u.speaker) ?? 0) + 1);
  return out;
}

/**
 * The voice match a line was voiced with: only what the engine reports, for the line or else for its speaker.
 * null for a stock-voice line, or when the engine reports none. Never the match that was asked for.
 */
export function lineMatch(u: Pick<DubUnit, "voice" | "cloneStrength">, speaker?: Pick<Speaker, "cloneStrength">): CloneStrength | null {
  if (u.voice !== "cloned") return null;
  return u.cloneStrength ?? speaker?.cloneStrength ?? null;
}

/** "Cloned", or "Cloned · <match>" when the engine reports the match the line was built with. */
export function clonedLabel(u: Pick<DubUnit, "voice" | "cloneStrength">, speaker?: Pick<Speaker, "cloneStrength">): string {
  const m = lineMatch(u, speaker);
  return m ? `Cloned · ${cloneStrengthLabel(m)}` : "Cloned";
}

/**
 * The Speakers card's "Voice match" row. It names the requested match and says whether the engine reports having
 * built the cloned voices with it; the row never claims a match the engine didn't report.
 */
export function matchSummary(requested: CloneStrength, speakers: readonly Speaker[]): { mode: string; note: string; warn: boolean; title: string } {
  const mode = cloneStrengthLabel(requested);
  const built = speakers.filter((s) => s.voice === "cloned");
  const reported = built.filter((s) => s.cloneStrength);
  if (!built.length) {
    return { mode, note: "requested", warn: false, title: "Asked for when this video was opened. Each voice shows the match it's built with once it's cloned." };
  }
  if (!reported.length) {
    return {
      mode, note: "requested, not applied yet", warn: true,
      title: "The engine hasn't reported building any voice with this match, so the cloned voices don't reflect it yet.",
    };
  }
  if (reported.length === built.length && reported.every((s) => s.cloneStrength === requested)) {
    return { mode, note: "", warn: false, title: "Every cloned voice is built with this match. Change it in Settings; it applies to the next video you open." };
  }
  return {
    mode, note: "requested, not used for every voice", warn: true,
    title: "Some cloned voices are built with another match, or the engine hasn't reported theirs. Each speaker shows the match it uses.",
  };
}

/**
 * A speaker's prepared lines that haven't started playing, in the given voice, and when the last of them ends.
 * The engine voices lines ahead of the playhead, so these keep the voice they were made with after a switch.
 */
export function preparedAhead(units: readonly DubUnit[], speaker: string, time: number, voice: DubUnit["voice"]): { lines: number; until: number } {
  let lines = 0, until = 0;
  for (const u of units) {
    if (u.speaker !== speaker || u.voice !== voice || u.start < time) continue;
    lines += 1;
    until = Math.max(until, u.start + u.audioWall);
  }
  return { lines, until };
}

const lines = (n: number) => `${n} prepared ${n === 1 ? "line" : "lines"}`;

/** The body of the "Use the preset voice?" confirmation: when the switch will actually be heard. */
export function presetConfirmBody(status: SpeakerStatus, ahead: { lines: number; until: number }): string {
  const what = status === "cloned"
    ? "Lines not yet prepared will be dubbed in a stock Telugu voice instead of their cloned voice, so they won't sound like themselves."
    : "Their lines will be dubbed in a stock Telugu voice even once their clone is ready, so they won't sound like themselves.";
  const up = fmtTime(ahead.until);
  const kept = ahead.lines === 0
    ? "Lines already dubbed don't change."
    : ahead.lines === 1
      ? `The line already prepared, up to ${up}, keeps the voice it was made with.`
      : `The ${ahead.lines} lines already prepared, up to ${up}, keep the voice they were made with.`;
  return `${what} ${kept} You can switch back any time; that too applies from the next line not yet prepared.`;
}

/**
 * The amber note under a speaker whose voice isn't (or isn't yet) what the panel shows: a preset switch that hasn't
 * reached the prepared lines yet, a stock-voice fallback, or lines already dubbed in the stock voice. null: none.
 */
export function voiceNote(s: Speaker, units: readonly DubUnit[], time: number): string | null {
  const why = presetReason(s);
  if (why === "chosen") {
    const keep = preparedAhead(units, s.id, time, "cloned");
    return keep.lines
      ? `Stock Telugu voice from ${fmtTime(keep.until)}: ${lines(keep.lines)} still ${keep.lines === 1 ? "uses" : "use"} the cloned voice`
      : "Stock Telugu voice, not a clone: you switched it on";
  }
  if (why === "fallback") return "Stock Telugu voice, not a clone: this speaker couldn't be cloned yet";
  const stock = preparedAhead(units, s.id, time, "preset");
  if (stock.lines && speakerStatus(s) === "cloned") {
    return `Cloned voice from ${fmtTime(stock.until)}: ${lines(stock.lines)} still ${stock.lines === 1 ? "uses" : "use"} the preset voice`;
  }
  const dubbed = presetLineCounts(units).get(s.id) ?? 0;
  return dubbed ? `${dubbed} ${dubbed === 1 ? "line" : "lines"} dubbed in the preset voice` : null;
}

/** A step of the preset switch: which speaker's confirmation is open, and what to send the engine, if anything. */
export type PresetStep = { confirming: string | null; send: { speaker: string; usePreset: boolean } | null };

/** A click on a speaker's preset switch: switching to the stock voice asks first; switching back is immediate. */
export function flipPreset(s: Pick<Speaker, "id" | "usePreset">, confirming: string | null): PresetStep {
  if (s.usePreset) return { confirming: null, send: { speaker: s.id, usePreset: false } };
  return { confirming: confirming === s.id ? null : s.id, send: null };
}

/** The confirmation's answer: "Use preset voice" switches; cancel (or Escape) changes nothing. */
export function settlePreset(id: string, usePreset: boolean): PresetStep {
  return { confirming: null, send: usePreset ? { speaker: id, usePreset: true } : null };
}
