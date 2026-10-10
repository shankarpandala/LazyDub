import type { FoundSpeaker, Render, VoiceProfile } from "./types";

export const VOICE_PROFILES: readonly VoiceProfile[] = ["auto", "male", "female"];
export const voiceProfileLabel = (profile: VoiceProfile): string =>
  profile === "auto" ? "Auto" : profile === "male" ? "Male" : "Female";

/** An acoustic suggestion names the dubbing voice, never the person's gender. */
export function voiceProfileDescription(speaker: FoundSpeaker): string {
  if (speaker.voiceProfile && speaker.voiceProfile !== "auto") {
    return `${voiceProfileLabel(speaker.voiceProfile)} voice selected`;
  }
  if (speaker.resolvedVoiceProfile && speaker.voiceProfileSource === "acoustic") {
    return `Auto suggestion: ${voiceProfileLabel(speaker.resolvedVoiceProfile)} voice`;
  }
  return "Auto: choose a voice";
}

export function voiceProfileSampleReady(speaker: FoundSpeaker): boolean {
  if (speaker.voiceCompatible === false) return false;
  if (speaker.voiceProfile === undefined) return true; // older engines use the card's existing provenance gate
  return speaker.resolvedVoiceProfile !== undefined && speaker.resolvedVoiceProfile !== null
    && (speaker.voiceProfile === "auto" || speaker.voiceProfile === speaker.resolvedVoiceProfile);
}

export function voiceProfileChangeCost(job: Render | null): string {
  if (job?.status === "running" || job?.status === "waiting") {
    return "Changing voice regenerates this speaker’s audio.";
  }
  if (job?.status === "queued") return "Voice choice applies when this queued dub runs. Saved video stays unchanged.";
  return "Voice choice applies when you resume or dub again. Saved video stays unchanged.";
}
