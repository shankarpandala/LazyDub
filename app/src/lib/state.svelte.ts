import type { DubUnit, Speaker, VideoInfo } from "./types";

export type Stage = "idle" | "resolving" | "fetching" | "listening" | "translating" | "voicing" | "ready" | "waiting" | "error";

export type Settings = {
  speedCap: number;
  allowSlowdown: boolean;
  allowFreeze: boolean;
  maxPause: number;
  style: "colloquial" | "formal";
  voiceMode: "clone" | "preset";
  updateCheck: boolean;
  hud: boolean;
  theme: "dark" | "light" | "system";
};

const DEFAULTS: Settings = {
  speedCap: 1.2, allowSlowdown: true, allowFreeze: true, maxPause: 1.5, style: "colloquial",
  voiceMode: "clone", updateCheck: true, hud: false, theme: "dark",
};

function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("maata.settings") ?? "{}") };
  } catch {
    return { ...DEFAULTS };
  }
}

export type Recent = { videoId: string; title: string; channel?: string; at: number; position: number };

function loadRecents(): Recent[] {
  try {
    return JSON.parse(localStorage.getItem("maata.recents") ?? "[]");
  } catch {
    return [];
  }
}

class AppState {
  connection = $state<"connecting" | "open" | "closed">("connecting");
  backend = $state("");
  device = $state("");
  demo = $state(false);

  video = $state<VideoInfo | null>(null);
  stage = $state<Stage>("idle");
  message = $state("");
  error = $state<{ message: string; retryable: boolean } | null>(null);

  units = $state<DubUnit[]>([]);
  skipped = $state<{ id: number; start: number; end: number }[]>([]);
  readyUntil = $state(0);
  speakers = $state<Speaker[]>([]);

  playing = $state(false);
  time = $state(0);
  duration = $state(0);
  speed = $state(1);
  volume = $state(1);
  waitingForDub = $state(false);

  settings = $state<Settings>(loadSettings());
  settingsOpen = $state(false);
  recents = $state<Recent[]>(loadRecents());
  windowSeconds = $state<number[]>([]);

  lead = $derived(Math.max(0, this.readyUntil - this.time));
  current = $derived(this.units.find((u) => this.time >= u.start && this.time <= u.start + u.audioWall + 0.3) ?? null);

  saveSettings(): void {
    localStorage.setItem("maata.settings", JSON.stringify(this.settings));
  }

  remember(): void {
    if (!this.video) return;
    const r: Recent = { videoId: this.video.videoId, title: this.video.title || "Untitled video", channel: this.video.channel, at: Date.now(), position: this.time };
    this.recents = [r, ...this.recents.filter((x) => x.videoId !== r.videoId)].slice(0, 8);
    localStorage.setItem("maata.recents", JSON.stringify(this.recents));
  }

  resetVideo(): void {
    this.video = null;
    this.units = [];
    this.skipped = [];
    this.readyUntil = 0;
    this.speakers = [];
    this.playing = false;
    this.time = 0;
    this.duration = 0;
    this.error = null;
    this.stage = "idle";
    this.message = "";
    this.waitingForDub = false;
  }
}

export const app = new AppState();
