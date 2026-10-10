import { describe, expect, it } from "vitest";
import { voiceProfileChangeCost, voiceProfileDescription, voiceProfileSampleReady } from "./voices";
import type { FoundSpeaker, Render } from "./types";

const speaker: FoundSpeaker = { id: "S1", label: "Speaker 1", talkSeconds: 1, share: 1, firstAt: 0, turns: 1, activity: [] };

describe("voice profile guidance", () => {
  it("distinguishes manual choices, acoustic suggestions and unresolved automatic choices", () => {
    expect(voiceProfileDescription({ ...speaker, voiceProfile: "female", resolvedVoiceProfile: "male" })).toBe("Female voice selected");
    expect(voiceProfileDescription({ ...speaker, voiceProfile: "auto", resolvedVoiceProfile: "male", voiceProfileSource: "acoustic" })).toBe("Auto suggestion: Male voice");
    expect(voiceProfileDescription({ ...speaker, voiceProfile: "auto", resolvedVoiceProfile: null })).toBe("Auto: choose a voice");
    expect(voiceProfileSampleReady({ ...speaker, voiceProfile: "female", resolvedVoiceProfile: "male" })).toBe(false);
    expect(voiceProfileSampleReady({ ...speaker, voiceProfile: "auto", resolvedVoiceProfile: null })).toBe(false);
  });

  it("never promises a paused, failed or completed job will restart after saving a choice", () => {
    for (const status of ["paused", "failed", "done"] as const) {
      expect(voiceProfileChangeCost({ status } as Render)).toBe("Voice choice applies when you resume or dub again. Saved video stays unchanged.");
    }
    expect(voiceProfileChangeCost({ status: "queued" } as Render)).toContain("when this queued dub runs");
    expect(voiceProfileChangeCost({ status: "running" } as Render)).toBe("Changing voice regenerates this speaker’s audio.");
  });
});
