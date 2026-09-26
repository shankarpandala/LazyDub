import type { Notice } from "./types";

const KINDS: readonly Notice["kind"][] = ["fast_speech"];

/** Seconds before a flagged stretch that its notice already shows, so the lag isn't a surprise. */
export const NOTICE_LEAD = 2;

/**
 * The notices with `n` added: overlapping or touching stretches of the same kind merge into one. A notice the UI
 * can't place (unknown kind, unreadable or empty span) is dropped.
 */
export function addNotice(list: readonly Notice[], n: Notice): Notice[] {
  if (!KINDS.includes(n.kind) || !Number.isFinite(n.from) || !Number.isFinite(n.to) || n.to <= n.from) return [...list];
  let merged: Notice = { kind: n.kind, from: n.from, to: n.to };
  const out: Notice[] = [];
  for (const x of list) {
    if (x.kind === merged.kind && x.from <= merged.to && merged.from <= x.to) {
      merged = { kind: x.kind, from: Math.min(x.from, merged.from), to: Math.max(x.to, merged.to) };
    } else {
      out.push(x);
    }
  }
  return [...out, merged].sort((a, b) => a.from - b.from);
}

/** The notice covering video time t (from NOTICE_LEAD s before its stretch to its end), if any. */
export function noticeAt(list: readonly Notice[], t: number): Notice | null {
  return list.find((n) => t >= n.from - NOTICE_LEAD && t < n.to) ?? null;
}

/** What the player's status chip says during a notice. */
export function noticeText(n: Notice): string {
  switch (n.kind) {
    case "fast_speech":
      return "Fast speech here · the dub lags slightly";
  }
}
