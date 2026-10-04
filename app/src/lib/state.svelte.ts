import type {
  ClaudeHealth, ClaudeProblem, ClientMessage, DubOptions, EngineMessage, EngineSettings, JobItem, Render, SpeakersFound,
  VideoInfo,
} from "./types";
import { PRIVACY_KEY, healthProblem } from "./claude";
import { jobOptions, newDubOptions } from "./settings";

/** Library (home), New dub (after a pasted link) and a job's progress (OFFLINE-RENDER §5). */
export type View = "library" | "new" | "job";

type Send = (m: ClientMessage) => void;

/** How often a pending removal asks the engine again while the job's pause finishes (ms), and for how long at most. */
export const REMOVE_RETRY_MS = 700;
export const REMOVE_GIVE_UP_MS = 120_000;
/** The engine's answer to a remove while the job still runs (server.py `_remove`). */
const PAUSE_FIRST = "Pause it first.";

function loadFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

/** A job that sends no `render` (it isn't running), as the job view reads it: its `renders` item. */
export function renderOf(item: JobItem): Render {
  return {
    videoId: item.videoId, status: item.status, stage: item.stage, stages: item.stages ?? [], stopAt: item.stopAt,
    finalUntil: item.finalUntil, eta: null, elapsed: item.elapsed ?? 0, position: item.position, output: item.output,
    slept: item.slept ?? null, coverage: item.coverage ?? null, report: item.report ?? null, error: item.error ?? null,
  };
}

/** The YouTube link of a video id: what `prepare` takes. */
export const youtuBe = (videoId: string) => `https://youtu.be/${videoId}`;

/**
 * The app's state: what the engine said (the jobs, their progress, the speakers found, the settings), what the user is
 * looking at, and nothing stored: the engine keeps the jobs and the one setting (§4).
 */
export class AppState {
  connection = $state<"connecting" | "open" | "closed">("connecting");
  backend = $state("");
  device = $state("");
  demo = $state(false);
  /** The Claude CLI as the engine found it on connect; null when the engine never calls Claude (the demo). */
  claude = $state<ClaudeHealth | null>(null);
  /** What holds translation back now, and when the engine said so (ms): the banner's countdown runs from then. */
  claudeProblem = $state<(ClaudeProblem & { at: number }) | null>(null);
  /** The one-time notice that transcript text goes to Anthropic has been read. */
  privacySeen = $state(loadFlag(PRIVACY_KEY));

  view = $state<View>("library");
  /** The library, newest activity first (`renders`), each job kept up to date by its `render` messages. */
  jobs = $state<JobItem[]>([]);
  /** The newest `render` of each job that sent one, by video id. */
  renders = $state<Record<string, Render>>({});
  /** The job the job view shows. */
  openId = $state<string | null>(null);
  /** The speaker check of each job that sent one (`speakers_found`), by video id. */
  foundBy = $state<Record<string, SpeakersFound>>({});
  /** The video New dub shows (`inspect`'s answer). */
  video = $state<VideoInfo | null>(null);
  /** The link `inspect` was sent for, until its answer. */
  inspecting = $state<string | null>(null);
  settings = $state<EngineSettings | null>(null);
  error = $state<{ message: string; retryable: boolean } | null>(null);
  settingsOpen = $state(false);
  /** Removals waiting for the job's pause to finish (the engine refuses to remove a running job): since, last try (ms). */
  removing = $state<Record<string, { since: number; last: number }>>({});
  /** A link to open in New dub once the engine is there (`?v=`, the shell's MAATA_OPEN). */
  launch: string | null = null;
  /** The video the user just asked to dub: its first `render` opens the job view. */
  private preparing: string | null = null;
  /** `renders` was asked for and hasn't come yet. */
  private asked = false;

  /** The open job: its newest `render`, else its `renders` item. */
  job = $derived.by((): Render | null => {
    const id = this.openId;
    if (!id) return null;
    const item = this.jobs.find((j) => j.videoId === id);
    return this.renders[id] ?? (item ? renderOf(item) : null);
  });
  /** The open job's `renders` item (title, thumbnail, options). */
  jobItem = $derived(this.jobs.find((j) => j.videoId === this.openId) ?? null);
  /**
   * The open job's speaker check, unless its speakers stage runs again (a Wrong count? being done): that check is of
   * the diarization going.
   */
  found = $derived.by((): SpeakersFound | null => {
    const f = this.openId ? this.foundBy[this.openId] : undefined;
    const speakers = this.job?.stages.find((s) => s.key === "speakers");
    return f && (!speakers || speakers.state === "done") ? f : null;
  });
  /** The job New dub's video already has, if any (fresher than `video.render`). */
  videoJob = $derived(this.video ? this.jobs.find((j) => j.videoId === this.video!.videoId) ?? this.video.render : null);
  /** The newest job's options: where New dub starts (§5). */
  lastOptions = $derived.by((): DubOptions => {
    const last = [...this.jobs].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0))[0];
    return newDubOptions(last?.settings);
  });

  /** What the engine said. `send` answers it where the UI needs more (the library, when a job it doesn't list moves). */
  receive(m: EngineMessage, send: Send, now: number = Date.now()): void {
    switch (m.type) {
      case "hello":
        this.backend = m.backend; this.device = m.device; this.demo = m.demo;
        this.setClaude(m.claude, now);
        this.jobs = m.renders ?? [];
        this.renders = m.render ? { [m.render.videoId]: m.render } : {};
        this.foundBy = {}; // (the engine sends each job's speaker check after hello)
        this.settings = m.settings ?? null;
        if (this.launch) {
          this.inspect(this.launch, send);
          this.launch = null;
        }
        break;
      case "video": {
        const { type: _t, url, ...video } = m;
        // Only the answer to the link pasted last: an earlier one still on its way (a link corrected) is dropped.
        if (this.inspecting !== null && (url === undefined || url === this.inspecting)) {
          this.video = video;
          this.inspecting = null;
          this.view = "new";
        }
        break;
      }
      case "render": {
        const { type: _t, ...r } = m;
        this.renders = { ...this.renders, [r.videoId]: r };
        const i = this.jobs.findIndex((j) => j.videoId === r.videoId);
        if (i >= 0) {
          const j = this.jobs[i]!;
          this.jobs[i] = {
            ...j, status: r.status, stage: r.stage, eta: r.eta, position: r.position, stopAt: r.stopAt,
            finalUntil: r.finalUntil, output: r.output, slept: r.slept, error: r.error, stages: r.stages,
            elapsed: r.elapsed, coverage: r.coverage, report: r.report ?? null,
          };
        } else if (!this.asked) {
          this.asked = true; // a job new to the library: its title, thumbnail and options come with `renders`
          send({ type: "renders" });
        }
        if (this.preparing === r.videoId) {
          this.preparing = null;
          this.openJob(r.videoId);
        }
        break;
      }
      case "renders": {
        this.asked = false;
        this.jobs = m.items;
        const status = new Map(m.items.map((j) => [j.videoId, j.status]));
        // A job that isn't running sends no `render` any more, so its item is the newer word (the queue moved up, its
        // MP4 was moved or deleted); a job no longer listed was removed, its speaker check with it.
        this.renders = Object.fromEntries(Object.entries(this.renders).filter(([id]) => {
          const s = status.get(id);
          return s === "running" || s === "waiting";
        }));
        this.foundBy = Object.fromEntries(Object.entries(this.foundBy).filter(([id]) => status.has(id)));
        for (const id of Object.keys(this.removing)) {
          if (!status.has(id)) this.forgetRemove(id);
        }
        if (this.openId && !status.has(this.openId) && this.view === "job") this.view = "library";
        break;
      }
      case "speakers_found": {
        const { type: _t, ...found } = m;
        this.foundBy = { ...this.foundBy, [found.videoId]: found };
        break;
      }
      case "settings":
        this.settings = { outputDir: m.outputDir };
        break;
      case "claude_error": {
        const { type: _t, ...problem } = m;
        this.claudeProblem = { ...problem, at: now };
        break;
      }
      case "claude_ok":
        this.claudeProblem = null;
        break;
      case "error":
        if (m.message === PAUSE_FIRST && Object.keys(this.removing).length) break; // the removal asks again
        if (m.url !== undefined && m.url !== this.inspecting) break; // a link pasted before the one asked about now
        this.error = { message: m.message, retryable: m.retryable };
        this.inspecting = null;
        this.preparing = null;
        break;
    }
  }

  /** Ask the engine about a pasted link; its `video` answer opens New dub. */
  inspect(url: string, send: Send): void {
    const u = url.trim();
    if (!u) return;
    this.error = null;
    this.inspecting = u;
    send({ type: "inspect", url: u });
  }

  /** New dub's Dub / Add to queue: the job runs, or waits its turn; its first `render` opens the job view. */
  dub(options: DubOptions, send: Send): void {
    if (!this.video) return;
    this.error = null;
    this.preparing = this.video.videoId;
    send({ type: "prepare", url: youtuBe(this.video.videoId), ...options });
  }

  openJob(videoId: string): void {
    this.openId = videoId;
    this.view = "job";
  }

  home(): void {
    this.view = "library";
  }

  pause(videoId: string, send: Send): void {
    send({ type: "pause", videoId });
  }

  resume(videoId: string, send: Send): void {
    send({ type: "resume", videoId });
  }

  /**
   * Go on with a job: a finished preview continues to the whole video (and any job with `whole`), and an expired one
   * is dubbed again with its own options, a preview as the preview it was (both a `prepare`); any other is resumed as
   * it was.
   */
  continueJob(item: Pick<JobItem, "videoId" | "status" | "output" | "settings" | "expired" | "stopAt">, send: Send,
              whole = false): void {
    const preview = !item.expired && item.status === "done" && (item.output?.kind === "preview" || item.stopAt != null);
    if (whole || preview || item.expired) {
      send({ type: "prepare", url: youtuBe(item.videoId), ...jobOptions(item.settings), ...(whole || preview ? { stopAt: null } : {}) });
    } else {
      send({ type: "resume", videoId: item.videoId });
    }
  }

  /**
   * Remove a job from the library (its MP4 stays). The engine refuses while it runs: it is paused first, and the
   * removal is asked again (`retryRemoves`) until its run has stopped.
   */
  remove(item: Pick<JobItem, "videoId" | "status">, send: Send, now: number = Date.now()): void {
    const status = this.renders[item.videoId]?.status ?? item.status;
    this.removing = { ...this.removing, [item.videoId]: { since: now, last: now } };
    if (status === "running" || status === "waiting") send({ type: "pause", videoId: item.videoId });
    else send({ type: "remove", videoId: item.videoId });
  }

  /** Called every REMOVE_RETRY_MS while removals are pending: ask again for the jobs whose run has stopped. */
  retryRemoves(send: Send, now: number = Date.now()): void {
    for (const [id, r] of Object.entries(this.removing)) {
      if (now - r.since > REMOVE_GIVE_UP_MS) {
        this.forgetRemove(id);
        this.error = { message: "The dub couldn't be removed: it didn't stop. Pause it, then remove it again.", retryable: true };
        continue;
      }
      const status = this.renders[id]?.status ?? this.jobs.find((j) => j.videoId === id)?.status;
      if (status === "running" || status === "waiting" || now - r.last < REMOVE_RETRY_MS) continue;
      this.removing = { ...this.removing, [id]: { ...r, last: now } };
      send({ type: "remove", videoId: id });
    }
  }

  private forgetRemove(id: string): void {
    const { [id]: _gone, ...rest } = this.removing;
    this.removing = rest;
  }

  /** The speaker check's "Wrong count?": the job runs its speakers again with this count. */
  setSpeakers(videoId: string, speakers: "auto" | number, send: Send): void {
    send({ type: "set_speakers", videoId, speakers });
    this.jobs = this.jobs.map((j) => (j.videoId === videoId ? { ...j, settings: { ...j.settings, speakers: speakers === "auto" ? null : speakers } } : j));
  }

  /** A speaker voiced with a stock voice instead of their clone, or back. Shown at once; the engine re-runs from there. */
  setVoice(videoId: string, speaker: string, usePreset: boolean, send: Send): void {
    send({ type: "set_voice", videoId, speaker, usePreset });
    this.jobs = this.jobs.map((j) => {
      if (j.videoId !== videoId) return j;
      const presets = new Set(j.settings?.presets ?? []);
      if (usePreset) presets.add(speaker); else presets.delete(speaker);
      return { ...j, settings: { ...j.settings, presets: [...presets].sort() } };
    });
  }

  openOutput(videoId: string, reveal: boolean, send: Send): void {
    send({ type: "open_output", videoId, reveal });
  }

  saveSettings(outputDir: string, send: Send): void {
    this.error = null;
    send({ type: "settings", outputDir: outputDir.trim() });
  }

  /** The engine's `hello`: how the Claude CLI is, and anything wrong with it already. */
  setClaude(health: ClaudeHealth | null | undefined, now: number = Date.now()): void {
    this.claude = health ?? null;
    const p = healthProblem(health);
    this.claudeProblem = p ? { ...p, at: now } : null;
  }

  ackPrivacy(): void {
    this.privacySeen = true;
    try {
      localStorage.setItem(PRIVACY_KEY, "1");
    } catch {
      // storage off: the notice shows again next time
    }
  }
}

export const app = new AppState();
