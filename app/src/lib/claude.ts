import type { ClaudeHealth, ClaudeProblem } from "./types";

/**
 * The Codex banner: a title, what it means for the dub, and what to do. The engine sends the problem; this turns
 * it into words. The internal Claude names are retained for compatibility with the engine protocol.
 */
export type ClaudeNotice = { title: string; detail: string; command?: string };

/** A new provider needs its own acknowledgment, even if the previous provider's notice was read. */
export const PRIVACY_KEY = "maata.codexPrivacySeen.v1";

export const PRIVACY_NOTICE = {
  title: "Translation goes through your Codex sign-in",
  detail:
    "Speech recognition and the Telugu voices run on this Mac, and audio never leaves it. Only text goes to OpenAI: " +
    "the English transcript and its translations, through your signed-in Codex CLI. Translation uses your ChatGPT " +
    "plan and counts toward that plan's usage.",
} as const;

const LIMITS: Record<string, string> = {
  session: "session", weekly: "weekly", overage: "extra-usage",
};

// Nothing plays in the app: a job held back by Codex does the work it can on this Mac meanwhile (OFFLINE-RENDER §4).
const KEEP = "The dub goes on with the work on this Mac meanwhile.";

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
 * The banner for a Codex problem. `retryIn` counts from when the engine sent it (`sentAt`, ms), so the countdown is
 * right however long the banner has been up.
 */
export function claudeNotice(p: ClaudeProblem, sentAt: number = Date.now(), now: number = Date.now()): ClaudeNotice {
  const left = p.retryIn == null ? null : Math.max(0, p.retryIn - (now - sentAt) / 1000);
  const retry = left == null ? "Maata tries again shortly." : left < 1 ? "Trying again now…" : `Trying again ${fmtRetry(left)}.`;
  switch (p.kind) {
    case "missing":
      return {
        title: "Codex CLI isn't installed",
        detail: `Install the Codex CLI from developers.openai.com/codex/cli, then sign in with your ChatGPT account. ${KEEP}`,
        command: "codex login",
      };
    case "not_signed_in":
      return {
        title: "Codex CLI isn't signed in",
        detail: `Sign in with your ChatGPT account in a terminal; Maata carries on by itself. ${KEEP}`,
        command: "codex login",
      };
    case "outdated":
      return {
        title: "Codex CLI needs an update",
        detail: `${p.message || "This version of Codex CLI is too old for Maata."} Update it using the method you installed it with. ${KEEP}`,
      };
    case "usage_limit": {
      const which = p.limit && LIMITS[p.limit] ? `${LIMITS[p.limit]} ` : "";
      const when = p.resetsAt ? `Translation resumes when it resets, at ${fmtReset(p.resetsAt, new Date(now))}.` : retry;
      return { title: `Your Codex ${which}usage limit is reached`, detail: `${when} ${KEEP}` };
    }
    case "stalled":
      return {
        title: "Codex CLI didn't start",
        detail: `Codex did not begin a response. ${retry} ${KEEP}`,
      };
    case "transient":
      return { title: "Codex is busy right now", detail: `${retry} ${KEEP}` };
    case "timeout":
      return { title: "Codex is taking too long to answer", detail: `${retry} ${KEEP}` };
    default:
      return { title: "Translation through Codex CLI failed", detail: `${p.message ? `${p.message} ` : ""}${retry} ${KEEP}` };
  }
}

/** The problem the engine found when the UI connected, if any. */
export function healthProblem(h: ClaudeHealth | null | undefined): ClaudeProblem | null {
  return h?.problem ? { kind: h.problem, message: h.message } : null;
}
