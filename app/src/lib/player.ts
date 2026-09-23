/**
 * Video players behind one interface. `YouTubePlayer` wraps the official IFrame API, always muted
 * (the original audio is never played). `DemoPlayer` draws a synthetic video so the app runs with
 * the demo engine, offline and in CI.
 */

export type PlayerState = "unstarted" | "playing" | "paused" | "buffering" | "ended";

export interface VideoPlayer {
  play(): void;
  pause(): void;
  seek(t: number): void;
  setRate(r: number): void;
  time(): number;
  duration(): number;
  rates(): number[];
  destroy(): void;
}

export type PlayerEvents = {
  state: (s: PlayerState) => void;
  rate: (r: number) => void;
  ready: () => void;
  error: (message: string) => void;
};

/* ---------------- YouTube IFrame player ---------------- */

type YTPlayer = {
  playVideo(): void;
  pauseVideo(): void;
  seekTo(t: number, allowSeekAhead: boolean): void;
  setPlaybackRate(r: number): void;
  getCurrentTime(): number;
  getDuration(): number;
  getAvailablePlaybackRates(): number[];
  mute(): void;
  destroy(): void;
};

declare global {
  interface Window {
    YT?: { Player: new (el: HTMLElement, opts: unknown) => YTPlayer; PlayerState: Record<string, number> };
    onYouTubeIframeAPIReady?: () => void;
  }
}

let apiLoading: Promise<void> | null = null;
function loadApi(): Promise<void> {
  if (window.YT?.Player) return Promise.resolve();
  apiLoading ??= new Promise((resolve, reject) => {
    window.onYouTubeIframeAPIReady = () => resolve();
    const s = document.createElement("script");
    s.src = "https://www.youtube.com/iframe_api";
    s.onerror = () => reject(new Error("Couldn't load the YouTube player."));
    document.head.appendChild(s);
  });
  return apiLoading;
}

const YT_ERRORS: Record<number, string> = {
  2: "YouTube rejected this video link.",
  5: "YouTube couldn't play this video here.",
  100: "This video was removed or is private.",
  101: "The owner doesn't allow this video to be embedded.",
  150: "The owner doesn't allow this video to be embedded.",
  152: "YouTube refused to play inside the app.",
  153: "YouTube refused to play inside the app.",
};

export async function createYouTubePlayer(host: HTMLElement, videoId: string, start: number, on: PlayerEvents): Promise<VideoPlayer> {
  await loadApi();
  const mount = document.createElement("div");
  host.appendChild(mount);
  let p!: YTPlayer;
  await new Promise<void>((resolve) => {
    p = new window.YT!.Player(mount, {
      host: "https://www.youtube-nocookie.com",
      videoId,
      width: "100%",
      height: "100%",
      playerVars: {
        autoplay: 0, controls: 0, disablekb: 1, fs: 0, rel: 0, playsinline: 1, iv_load_policy: 3,
        modestbranding: 1, start: Math.floor(start), mute: 1, origin: location.origin,
      },
      events: {
        onReady: () => {
          p.mute();
          on.ready();
          resolve();
        },
        onStateChange: (e: { data: number }) => {
          const S = window.YT!.PlayerState;
          const map: Record<number, PlayerState> = { [S.PLAYING!]: "playing", [S.PAUSED!]: "paused", [S.BUFFERING!]: "buffering", [S.ENDED!]: "ended" };
          on.state(map[e.data] ?? "unstarted");
          p.mute();
        },
        onPlaybackRateChange: (e: { data: number }) => on.rate(e.data),
        onError: (e: { data: number }) => on.error(YT_ERRORS[e.data] ?? "The YouTube player hit an error."),
      },
    });
  });
  return {
    play: () => p.playVideo(),
    pause: () => p.pauseVideo(),
    seek: (t) => p.seekTo(t, true),
    setRate: (r) => p.setPlaybackRate(r),
    time: () => p.getCurrentTime(),
    duration: () => p.getDuration(),
    rates: () => p.getAvailablePlaybackRates(),
    destroy: () => p.destroy(),
  };
}

/* ---------------- Demo player ---------------- */

export function createDemoPlayer(canvas: HTMLCanvasElement, total: number, start: number, on: PlayerEvents): VideoPlayer {
  const ctx = canvas.getContext("2d")!;
  let t = start;
  let rate = 1;
  let playing = false;
  let last = performance.now();
  let raf = 0;
  const draw = (now: number) => {
    const dt = (now - last) / 1000;
    last = now;
    if (playing) {
      t = Math.min(total, t + dt * rate);
      if (t >= total) {
        playing = false;
        on.state("ended");
      }
    }
    const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
    const h = (canvas.height = canvas.clientHeight * devicePixelRatio);
    const g = ctx.createLinearGradient(0, 0, w, h);
    const hue = (t * 6) % 360;
    g.addColorStop(0, `hsl(${(hue + 250) % 360} 45% 14%)`);
    g.addColorStop(1, `hsl(${(hue + 300) % 360} 55% 22%)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 3; i++) {
      const x = w * (0.5 + 0.32 * Math.sin(t * 0.35 + i * 2.1));
      const y = h * (0.5 + 0.28 * Math.cos(t * 0.27 + i * 1.7));
      const r = Math.min(w, h) * (0.22 + 0.05 * i);
      const rg = ctx.createRadialGradient(x, y, 0, x, y, r);
      rg.addColorStop(0, ["rgba(247,183,51,.35)", "rgba(242,96,60,.3)", "rgba(216,51,107,.28)"][i]!);
      rg.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = rg;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.fillStyle = "rgba(255,255,255,.12)";
    ctx.font = `${Math.round(h * 0.05)}px Inter Variable, sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText(`DEMO · ${t.toFixed(2)}s`, h * 0.05, h * 0.93);
    raf = requestAnimationFrame(draw);
  };
  raf = requestAnimationFrame(draw);
  queueMicrotask(() => on.ready());
  return {
    play: () => { if (!playing) { playing = true; last = performance.now(); on.state("playing"); } },
    pause: () => { if (playing) { playing = false; on.state("paused"); } },
    seek: (x) => { t = Math.max(0, Math.min(total, x)); on.state(playing ? "playing" : "paused"); },
    setRate: (r) => { rate = r; on.rate(r); },
    time: () => t,
    duration: () => total,
    rates: () => [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
    destroy: () => cancelAnimationFrame(raf),
  };
}
