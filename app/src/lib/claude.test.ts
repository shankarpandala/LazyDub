import { describe, expect, it } from "vitest";
import { PRIVACY_NOTICE, claudeNotice, fmtReset, fmtRetry, healthProblem } from "./claude";
import type { ClaudeProblem, ClaudeProblemKind } from "./types";

const at = (kind: ClaudeProblemKind, over: Partial<ClaudeProblem> = {}): ClaudeProblem => ({ kind, message: "", ...over });

describe("the Claude banner", () => {
  it("tells the user what to run when the CLI isn't signed in, installed or new enough", () => {
    expect(claudeNotice(at("not_signed_in"))).toMatchObject({ title: "Claude Code isn't signed in", command: "claude auth login" });
    expect(claudeNotice(at("missing")).title).toBe("Claude Code isn't installed");
    const old = claudeNotice(at("outdated", { message: "Claude Code 2.1.201 can't run claude-opus-5-5." }));
    expect(old).toMatchObject({ title: "Claude Code needs an update", command: "claude update" });
    expect(old.detail).toContain("2.1.201");
  });

  it("names the usage limit and when translation resumes", () => {
    const resets = Date.UTC(2026, 8, 25, 15, 0) / 1000;
    const n = claudeNotice(at("usage_limit", { limit: "session", resetsAt: resets }), 0, resets * 1000 - 3_600_000);
    expect(n.title).toBe("Your Claude 5-hour usage limit is reached");
    expect(n.detail).toContain(`resumes when it resets, at ${fmtReset(resets, new Date(resets * 1000 - 3_600_000))}`);
    expect(claudeNotice(at("usage_limit", { limit: "opus" })).title).toBe("Your Claude weekly Opus usage limit is reached");
    expect(claudeNotice(at("usage_limit")).title).toBe("Your Claude usage limit is reached");
  });

  it("names the likely cause when Claude Code didn't start (#91987) and counts down to the retry", () => {
    const n = claudeNotice(at("stalled", { retryIn: 60 }), 10_000, 25_000);
    expect(n.title).toBe("Claude Code didn't start");
    expect(n.detail).toContain("open in another window on the same version");
    expect(n.detail).toContain("#91987");
    expect(n.detail).toContain("Trying again in 45 s.");
    expect(claudeNotice(at("transient", { retryIn: 30 }), 0, 40_000).detail).toContain("Trying again now");
    expect(claudeNotice(at("failed", { message: "exit 3", retryIn: 240 }), 0, 0).detail).toContain("exit 3 Trying again in 4 min.");
  });

  it("always says the dub goes on with what is translated", () => {
    const kinds: ClaudeProblemKind[] = ["missing", "not_signed_in", "outdated", "usage_limit", "transient", "stalled", "timeout",
      "bad_output", "failed"];
    for (const k of kinds) expect(claudeNotice(at(k)).detail).toContain("Lines already translated keep playing.");
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
    const ok = { installed: true, version: "2.1.281", signedIn: true, models: [], model: "claude-sonnet-5", problem: null, message: "" };
    expect(healthProblem(ok)).toBeNull();
    expect(healthProblem(null)).toBeNull();
    expect(healthProblem({ ...ok, installed: false, problem: "missing", message: "not found" }))
      .toEqual({ kind: "missing", message: "not found" });
  });
  it("says what leaves the Mac, through whose account, and where the privacy setting is", () => {
    expect(PRIVACY_NOTICE.detail).toContain("audio never leaves");
    expect(PRIVACY_NOTICE.detail).toMatch(/transcript, as text, is sent to Anthropic by your own Claude Code/);
    expect(PRIVACY_NOTICE.detail).toContain("your own plan");
    expect(PRIVACY_NOTICE.detail).toContain("claude.ai");
  });
});
