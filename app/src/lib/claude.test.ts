import { describe, expect, it } from "vitest";
import { PRIVACY_NOTICE, claudeNotice, fmtReset, fmtRetry, healthProblem } from "./claude";
import type { ClaudeProblem, ClaudeProblemKind } from "./types";

const at = (kind: ClaudeProblemKind, over: Partial<ClaudeProblem> = {}): ClaudeProblem => ({ kind, message: "", ...over });

describe("the Codex banner", () => {
  it("explains how to sign in, install or update the CLI", () => {
    expect(claudeNotice(at("not_signed_in"))).toMatchObject({ title: "Codex CLI isn't signed in", command: "codex login" });
    expect(claudeNotice(at("missing"))).toMatchObject({ title: "Codex CLI isn't installed", command: "codex login" });
    const old = claudeNotice(at("outdated", { message: "Codex CLI cannot run the requested model." }));
    expect(old.title).toBe("Codex CLI needs an update");
    expect(old.detail).toContain("cannot run the requested model");
    expect(old.detail).toContain("Update it using the method you installed it with");
    expect(old.command).toBeUndefined();
  });

  it("names the usage limit and when translation resumes", () => {
    const resets = Date.UTC(2026, 8, 25, 15, 0) / 1000;
    const n = claudeNotice(at("usage_limit", { limit: "session", resetsAt: resets }), 0, resets * 1000 - 3_600_000);
    expect(n.title).toBe("Your Codex session usage limit is reached");
    expect(n.detail).toContain(`resumes when it resets, at ${fmtReset(resets, new Date(resets * 1000 - 3_600_000))}`);
    expect(claudeNotice(at("usage_limit", { limit: "weekly" })).title).toBe("Your Codex weekly usage limit is reached");
    expect(claudeNotice(at("usage_limit", { limit: "opus" })).title).toBe("Your Codex usage limit is reached");
    expect(claudeNotice(at("usage_limit")).title).toBe("Your Codex usage limit is reached");
  });

  it("reports a stalled CLI without inventing a cause and counts down to the retry", () => {
    const n = claudeNotice(at("stalled", { retryIn: 60 }), 10_000, 25_000);
    expect(n.title).toBe("Codex CLI didn't start");
    expect(n.detail).toContain("Codex did not begin a response");
    expect(n.detail).not.toContain("another window");
    expect(n.detail).not.toContain("#91987");
    expect(n.detail).toContain("Trying again in 45 s.");
    expect(claudeNotice(at("transient", { retryIn: 30 }), 0, 40_000).detail).toContain("Trying again now");
    expect(claudeNotice(at("failed", { message: "exit 3", retryIn: 240 }), 0, 0).detail).toContain("exit 3 Trying again in 4 min.");
  });

  it("always says the dub goes on with the work on this Mac", () => {
    const kinds: ClaudeProblemKind[] = ["missing", "not_signed_in", "outdated", "usage_limit", "transient", "stalled", "timeout",
      "bad_output", "failed"];
    for (const k of kinds) expect(claudeNotice(at(k)).detail).toContain("The dub goes on with the work on this Mac meanwhile.");
  });
});

describe("retry and reset times", () => {
  it("reads as seconds under a minute and a half, else minutes", () => {
    expect(fmtRetry(0.2)).toBe("in 1 s");
    expect(fmtRetry(89)).toBe("in 89 s");
    expect(fmtRetry(150)).toBe("in 3 min");
  });
  it("shows only the time for a reset later today, and the day too otherwise", () => {
    const now = new Date(2026, 8, 25, 9, 0);
    const later = new Date(2026, 8, 25, 15, 0).getTime() / 1000;
    const tomorrow = new Date(2026, 8, 26, 15, 0).getTime() / 1000;
    expect(fmtReset(later, now)).toBe(new Date(later * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }));
    expect(fmtReset(tomorrow, now).endsWith(fmtReset(later, now))).toBe(true);
    expect(fmtReset(tomorrow, now).length).toBeGreaterThan(fmtReset(later, now).length);
  });
});

describe("hello and privacy", () => {
  it("turns the CLI's state at connect into a banner only when something is wrong", () => {
    const ok = { installed: true, version: "1.0.0", signedIn: true, models: [], model: "gpt-6-luna", problem: null, message: "" };
    expect(healthProblem(ok)).toBeNull();
    expect(healthProblem(null)).toBeNull();
    expect(healthProblem({ ...ok, installed: false, problem: "missing", message: "not found" }))
      .toEqual({ kind: "missing", message: "not found" });
  });
  it("says which text leaves the Mac and whose plan it uses", () => {
    expect(PRIVACY_NOTICE.detail).toContain("audio never leaves");
    expect(PRIVACY_NOTICE.detail).toContain("Only text goes to OpenAI");
    expect(PRIVACY_NOTICE.detail).toContain("the English transcript and its translations");
    expect(PRIVACY_NOTICE.detail).toContain("through your signed-in Codex CLI");
    expect(PRIVACY_NOTICE.detail).toContain("your ChatGPT plan");
    expect(PRIVACY_NOTICE.detail).toContain("counts toward that plan's usage");
    expect(PRIVACY_NOTICE.detail).not.toMatch(/Claude|Anthropic/);
  });
});
