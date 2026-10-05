/**
 * What the job views say (OFFLINE-RENDER §5): the stepper's groups, the status chips, the ETA and elapsed time, the
 * queue position and New dub's estimate line. Pure, so it is unit-tested.
 */
import { fmtReset } from "./claude";
import { fmtBytes, fmtCount, fmtDuration, fmtSpan, fmtTime } from "./format";
import type {
  Coverage, Estimate, FoundSpeaker, JobItem, JobStatus, Render, SpeakersFound, StageKey, StageRow, StageState,
} from "./types";

/** The engine's stage labels (`render.STAGES`), for a job of the Library, which carries only its stage's key. */
export const STAGE_LABELS: Readonly<Record<StageKey, string>> = {
  fetch: "Download", speakers: "Speakers", transcript: "Transcript", units: "Sentences", voices: "Voices",
  brief: "Video brief", separate: "Background sound", translate: "Translation", voice_lines: "Telugu speech",
  finish: "Finishing", video: "Video download", export: "Saving the video",
};

/** The job view's stepper: the engine's stages as the user thinks of them, in the order they finish. */
export const STAGE_GROUPS: readonly { label: string; keys: readonly StageKey[] }[] = [
  { label: "Download", keys: ["fetch"] },
  { label: "Speakers", keys: ["speakers"] },
  { label: "Transcript", keys: ["transcript", "units"] },
  { label: "Voices & brief", keys: ["voices", "brief"] },
  { label: "Background sound", keys: ["separate"] },
  { label: "Translation", keys: ["translate"] },
  { label: "Telugu speech", keys: ["voice_lines"] },
  { label: "Finishing", keys: ["finish"] },
  { label: "Video download", keys: ["video"] },
  { label: "Saving the video", keys: ["export"] },
];

/** The speakers stage's progress from which pyannote is clustering, which has no count (torch_common's hook). */
export const CLUSTERING_FROM = 0.9;
export const CLUSTERING = "Grouping the voices";

export type GroupRow = {
  label: string;
  state: StageState;
  /** Its count in its own unit ("52 / 162 MB", "1:02:10 / 2:55:06"); "" when there is nothing to count. */
  count: string;
  /** done / total, 0–1, for the running row's bar; null when unknown. */
  share: number | null;
  /** The bar moves with no count: pyannote clustering, or a download of unknown size. */
  indeterminate: boolean;
  /** Seconds it took (its stages side by side: the longest). */
  seconds: number;
  /** Seconds it has left while the job runs, else null. */
  eta: number | null;
  /** What it is doing besides counting ("Waiting for the background sound", "Finishing the file…"). */
  note: string;
};

function groupState(rows: readonly StageRow[]): StageState {
  if (rows.some((r) => r.state === "failed")) return "failed";
  if (rows.some((r) => r.state === "running")) return "running";
  if (rows.length && rows.every((r) => r.state === "done")) return "done";
  return "todo";
}

const plural = (n: number, one: string, many = `${one}s`) => `${fmtCount(n)} ${n === 1 ? one : many}`;

/** A stage's count in its unit: done of total while it runs, the total once done, nothing before. */
export function countText(r: StageRow): string {
  if (r.state === "todo") return "";
  const done = r.state === "done";
  const of = (a: string, b: string) => (done ? b : `${a} / ${b}`);
  switch (r.unit) {
    case "MB":
      return r.total > 0 ? `${of(fmtCount(r.done), fmtCount(r.total))} MB` : `${fmtCount(r.done)} MB`;
    case "fraction":
      return done ? "" : `${Math.round(Math.min(Math.max(r.done, 0), 1) * 100)} %`;
    case "video s":
    case "speech s":
      return r.total > 0 ? of(fmtTime(r.done), fmtTime(r.total)) : "";
    case "lines":
    case "speakers":
    case "parts": {
      const one = r.unit.slice(0, -1);
      return done ? plural(r.total, one) : `${fmtCount(r.done)} / ${plural(r.total, one)}`;
    }
    default:
      return r.total > 0 ? of(fmtCount(r.done), fmtCount(r.total)) : "";
  }
}

/** The stepper's rows from a `render`'s stages. */
export function groupRows(stages: readonly StageRow[]): GroupRow[] {
  const byKey = new Map(stages.map((s) => [s.key, s]));
  return STAGE_GROUPS.map(({ label, keys }) => {
    const rows = keys.map((k) => byKey.get(k)).filter((r): r is StageRow => r !== undefined);
    const state = groupState(rows);
    const lead = rows.find((r) => r.state === "running") ?? rows.find((r) => r.state === "failed") ?? rows[0];
    const clustering = lead?.key === "speakers" && lead.state === "running" && lead.done >= CLUSTERING_FROM;
    const etas = rows.filter((r) => r.state !== "done" && r.eta != null).map((r) => r.eta!);
    const share = lead && lead.total > 0 ? Math.min(Math.max(lead.done / lead.total, 0), 1) : null;
    return {
      label, state,
      count: lead && !clustering ? countText(lead) : "",
      share: clustering ? null : share,
      indeterminate: state === "running" && (clustering || share === null),
      seconds: Math.max(0, ...rows.map((r) => r.seconds || 0)),
      eta: etas.length ? Math.max(...etas) : null,
      note: clustering ? CLUSTERING : rows.find((r) => r.state === "running" && r.extra)?.extra ?? "",
    };
  });
}

/** "next in line", "2nd in line", "23rd in line". */
export function positionText(n: number | null | undefined): string {
  if (!n || n < 1) return "in line";
  if (n === 1) return "next in line";
  const teen = n % 100 >= 11 && n % 100 <= 13;
  const suffix = teen ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix} in line`;
}

/** "about 2 h 10 min left"; "" when unknown. */
export function etaText(eta: number | null | undefined): string {
  return eta == null || !Number.isFinite(eta) ? "" : `about ${fmtDuration(eta)} left`;
}

/** "1 h 5 min so far"; with `done`, "took 3 h 10 min". */
export function elapsedText(elapsed: number | null | undefined, done = false): string {
  return !elapsed ? "" : done ? `took ${fmtDuration(elapsed)}` : `${fmtDuration(elapsed)} so far`;
}

export type Tone = "run" | "wait" | "ok" | "warn" | "danger" | "muted";
export type Chip = { text: string; tone: Tone };

/** For how long after the Mac woke a running job's chip says it slept (s). */
export const SLEEP_NOTE_FOR = 15 * 60;

/** A job as the chip reads it: its `renders` item, with the newest `render` of it over it when there is one. */
export type ChipJob = Pick<JobItem, "status" | "position" | "stage" | "eta" | "duration" | "stopAt" | "output" | "expired">
  & { slept?: JobItem["slept"]; error?: string | null; stages?: readonly StageRow[] };

/** The job with its newest `render` laid over its `renders` item. */
export function live(item: JobItem, r: Render | null | undefined): ChipJob {
  if (!r || r.videoId !== item.videoId) return item;
  return {
    ...item, status: r.status, position: r.position, stage: r.stage, eta: r.eta, stopAt: r.stopAt, output: r.output,
    slept: r.slept, error: r.error, stages: r.stages,
  };
}

/** Whether the job is running or about to (it holds the GPU, or waits for it). */
export const isActive = (s: JobStatus) => s === "running" || s === "waiting" || s === "queued";

/** The sleep note: "Paused while the Mac slept · resumed 06:40", while it is news (SLEEP_NOTE_FOR). */
export function sleptText(slept: JobItem["slept"] | undefined, now: number = Date.now()): string {
  if (!slept || now / 1000 - slept.resumed > SLEEP_NOTE_FOR) return "";
  return `Paused while the Mac slept · resumed ${fmtReset(slept.resumed, new Date(now))}`;
}

/**
 * The Library's status chip (§5). `now`: ms; `resetsAt`: when the usage limit holding Codex back resets (epoch s).
 */
export function chip(job: ChipJob, now: number = Date.now(), resetsAt: number | null = null): Chip {
  const s = job.status;
  if (job.expired && !isActive(s)) return { text: "Expired · Dub again to re-voice", tone: "muted" };
  switch (s) {
    case "queued":
      return { text: `Queued · ${positionText(job.position)}`, tone: "wait" };
    case "running": {
      const slept = sleptText(job.slept, now);
      if (slept) return { text: slept, tone: "wait" };
      const at = job.stages?.find((r) => r.key === job.stage);
      const share = !at ? null : at.unit === "fraction" ? at.done : at.total > 0 ? at.done / at.total : null;
      const pct = share == null ? "" : ` ${Math.floor(100 * Math.min(Math.max(share, 0), 1))} %`;
      const stage = job.stage ? `${STAGE_LABELS[job.stage] ?? job.stage}${pct}` : "";
      return { text: ["Dubbing", stage, etaText(job.eta)].filter(Boolean).join(" · "), tone: "run" };
    }
    case "waiting":
      return { text: resetsAt ? `Waiting for Codex · resets ${fmtReset(resetsAt, new Date(now))}` : "Waiting for Codex", tone: "warn" };
    case "paused":
      return { text: job.stage ? `Paused at ${STAGE_LABELS[job.stage] ?? job.stage}` : "Paused", tone: "muted" };
    case "interrupted":
      return { text: "Continuing after Maata was closed", tone: "wait" };
    case "failed":
      return { text: `Failed · ${job.error || "something went wrong"}`, tone: "danger" };
    case "done": {
      const out = job.output;
      if (out?.missing) return { text: "File moved or deleted", tone: "warn" };
      if (out?.kind === "preview" || (!out && job.stopAt)) {
        return { text: `Preview done · first ${Math.round((job.stopAt ?? 900) / 60)} min`, tone: "ok" };
      }
      return { text: ["Done", fmtTime(job.duration), out ? fmtBytes(out.bytes) : ""].filter(Boolean).join(" · "), tone: "ok" };
    }
    default:
      return { text: String(s), tone: "muted" };
  }
}

/** How many jobs would run before a new one for `videoId`: the one running and those queued. */
export function jobsAhead(jobs: readonly Pick<JobItem, "videoId" | "status" | "position">[], videoId: string): number {
  return jobs.filter((j) => j.videoId !== videoId && (isActive(j.status) || j.position != null)).length;
}

/** "Movies ▸ Maata" for ~/Movies/Maata; other folders by their parts, the last three at most. */
export function folderText(path: string): string {
  const rel = path.replace(/^\/(?:Users|home)\/[^/]+\/?/, "");
  const parts = (rel || path).split(/[\\/]/).filter(Boolean);
  if (!parts.length) return path;
  return (parts.length > 3 ? ["…", ...parts.slice(-2)] : parts).join(" ▸ ");
}

/**
 * New dub's estimate line (§5): "About 3–6½ h on this Mac · about 310–360 Codex calls · saves to Movies ▸ Maata", and
 * "after the 1 job ahead (about 2 h)" when it would be queued. `claude`: the engine translates through Codex (the demo
 * doesn't).
 */
export function estimateLine(est: Estimate, opts: {
  stopAt: number | null; outputDir: string | null; ahead: number; claude: boolean;
}): string {
  const e = opts.stopAt && est.preview ? est.preview : est;
  const parts = [`About ${fmtSpan(e.seconds[0], e.seconds[1])} on this Mac`];
  if (opts.claude) {
    const [lo, hi] = e.claudeCalls;
    parts.push(`about ${lo === hi ? fmtCount(hi) : `${fmtCount(lo)}–${fmtCount(hi)}`} Codex calls`);
  }
  if (opts.outputDir) parts.push(`saves to ${folderText(opts.outputDir)}`);
  if (opts.ahead > 0) {
    const jobs = opts.ahead === 1 ? "the 1 job ahead" : `the ${opts.ahead} jobs ahead`;
    parts.push(est.ahead ? `after ${jobs} (about ${fmtDuration(est.ahead)})` : `after ${jobs}`);
  }
  return parts.join(" · ");
}

/** The coverage classes of the wordings voiced, as shares of the lines classed (best first); [] before any. */
export function coverageShares(c: Coverage | null | undefined): { key: string; label: string; share: number }[] {
  if (!c) return [];
  const rows = [
    { key: "C", label: "Complete", n: c.C }, { key: "m", label: "Minor changes", n: c.m },
    { key: "P", label: "Partly", n: c.P }, { key: "E", label: "With an error", n: c.E },
  ];
  const total = rows.reduce((a, r) => a + (r.n || 0), 0);
  return total ? rows.filter((r) => r.n > 0).map(({ key, label, n }) => ({ key, label, share: n / total })) : [];
}

const row = (r: Render | null | undefined, key: StageKey) => r?.stages.find((s) => s.key === key);

/** The Telugu speech has started: from here a change of speakers or voices costs speech synthesis. */
export function voicingStarted(r: Render | null | undefined): boolean {
  const v = row(r, "voice_lines");
  return !!v && (v.state === "done" || v.done > 0);
}

/** Seconds until the Telugu speech starts, while the job runs (the stages before it, side by side as they run). */
export function untilVoicing(r: Render | null | undefined): number | null {
  if (!r || (r.status !== "running" && r.status !== "waiting")) return null;
  const eta = (k: StageKey) => {
    const s = row(r, k);
    return s && s.state !== "done" ? s.eta ?? 0 : 0;
  };
  return eta("speakers") + eta("transcript") + eta("units") + Math.max(eta("voices"), eta("brief")) + eta("separate");
}

/** What re-running the speakers costs now (§5). */
export function speakerChangeCost(r: Render | null | undefined): string {
  if (voicingStarted(r)) return "Lines whose speaker changes are translated and voiced again.";
  const left = untilVoicing(r);
  return left ? `Free until the background sound is done, in about ${fmtDuration(left)}.` : "Free now: nothing is voiced yet.";
}

/** What switching a speaker to or from the stock voice costs now. */
export function voiceSwitchCost(r: Render | null | undefined, label: string): string {
  return voicingStarted(r) ? `${label}'s lines are voiced again.` : "Free now: nothing is voiced yet.";
}

/** "Found 2 speakers", and the merges: "1 short voice merged into Speaker 1". */
export function foundText(f: SpeakersFound): { title: string; merges: string[] } {
  const label = (id: string) => f.speakers.find((s: FoundSpeaker) => s.id === id)?.label ?? `Speaker ${id.replace(/\D/g, "")}`;
  const groups = new Map<string, number>();
  for (const m of f.merged) {
    const key = `${m.why === "same" ? "same" : "talk"}|${m.into}`;
    groups.set(key, (groups.get(key) ?? 0) + 1);
  }
  const merges = [...groups].map(([key, n]) => {
    const [why, into] = key.split("|") as [string, string];
    const what = why === "same" ? (n === 1 ? "matching voice" : "matching voices") : (n === 1 ? "short voice" : "short voices");
    return `${n} ${what} merged into ${label(into)}`;
  });
  return { title: `Found ${plural(f.speakers.length, "speaker")}`, merges };
}
