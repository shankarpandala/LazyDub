import type { ClaudeHealth, ClaudeProblem } from "./types";

/**
 * What the Claude banner says (ADR-019, ARCHITECTURE §3.12): a title, what it means for the dub, and what to do. The
 * engine sends the problem; this turns it into words. Pure, so it is unit-tested.
 */
export type ClaudeNotice = { title: string; detail: string; command?: string };

/** localStorage key: the one-time notice that transcript text goes to Anthropic has been read. */
export const PRIVACY_KEY = "maata.claudePrivacySeen";

export const PRIVACY_NOTICE = {
  title: "Translation goes through your Claude Code",
  detail:
    "Speech recognition and the Telugu voices run on this Mac, and audio never leaves it. The English transcript, as text, " +
    "is sent to Anthropic by your own Claude Code, signed in with your own plan, and counts toward that plan's usage. " +
    "Whether Anthropic may use it to improve its models, and how long it is kept, follow the privacy settings of your " +
    "account on claude.ai.",
} as const;

const LIMITS: Record<string, string> = {
  session: "5-hour", weekly: "weekly", opus: "weekly Opus", sonnet: "weekly Sonnet", overage: "extra-usage",
};

const KEEP = "Lines already translated keep playing.";

/** "3:05 PM", or "Tue 7 Oct, 2:00 AM" when it isn't today (the viewer's own locale and time zone). */
export function fmtReset(resetsAt: number, now: Date = new Date()): string {
  const at = new Date(resetsAt * 1000);
  const time = at.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (at.toDateString() === now.toDateString()) return time;
  return `${at.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}, ${time}`;
}

/** "in 45 s", "in 3 min". */
export function fmtRetry(seconds: number): string {
  return seconds < 90 ? `in ${Math.max(1, Math.round(seconds))} s` : `in ${Math.round(seconds / 60)} min`;
}

/**
 * The banner for a Claude problem. `retryIn` counts from when the engine sent it (`sentAt`, ms), so the countdown is
 * right however long the banner has been up.
 */
export function claudeNotice(p: ClaudeProblem, sentAt: number = Date.now(), now: number = Date.now()): ClaudeNotice {
  const left = p.retryIn == null ? null : Math.max(0, p.retryIn - (now - sentAt) / 1000);
  const retry = left == null ? "Maata tries again shortly." : left < 1 ? "Trying again now…" : `Trying again ${fmtRetry(left)}.`;
  switch (p.kind) {
    case "missing":
      return {
        title: "Claude Code isn't installed",
        detail: `Maata translates through your own Claude Code: install it from claude.com/claude-code, then sign in. ${KEEP}`,
        command: "claude auth login",
      };
    case "not_signed_in":
      return {
        title: "Claude Code isn't signed in",
        detail: `Sign in with your own Claude plan in a terminal; Maata carries on by itself. ${KEEP}`,
        command: "claude auth login",
      };
    case "outdated":
      return {
        title: "Claude Code needs an update",
        detail: `${p.message || "This version of Claude Code is too old for Maata."} ${KEEP}`,
        command: "claude update",
      };
    case "usage_limit": {
      const which = p.limit ? `${LIMITS[p.limit] ?? p.limit} ` : "";
      const when = p.resetsAt ? `Translation resumes when it resets, at ${fmtReset(p.resetsAt, new Date(now))}.` : retry;
      return { title: `Your Claude ${which}usage limit is reached`, detail: `${when} ${KEEP}` };
    }
    case "stalled":
      return {
        title: "Claude Code didn't start",
        detail:
          "If Claude Code is open in another window on the same version, it can block Maata (known issue #91987). " +
          `${retry} ${KEEP}`,
      };
    case "transient":
      return { title: "Claude is busy right now", detail: `${retry} ${KEEP}` };
    case "timeout":
      return { title: "Claude is taking too long to answer", detail: `${retry} ${KEEP}` };
    default:
      return { title: "Translation through Claude Code failed", detail: `${p.message ? `${p.message} ` : ""}${retry} ${KEEP}` };
  }
}

/** The problem the engine found when the UI connected, if any. */
export function healthProblem(h: ClaudeHealth | null | undefined): ClaudeProblem | null {
  return h?.problem ? { kind: h.problem, message: h.message } : null;
}
