import { expect, it } from "vitest";
import { hear } from "./hear.svelte";

it("cancels a pending sample so a late old-voice frame cannot start after changing the profile", () => {
  hear.waiting = "S1";
  hear.stop();
  expect(hear.waiting).toBeNull();
  // Node has no AudioContext: a stale frame must return before trying to create one.
  expect(() => hear.play("S1", { id: 1, samples: new Float32Array([0.1]), sampleRate: 24000 })).not.toThrow();
  expect(hear.playing).toBeNull();
});
