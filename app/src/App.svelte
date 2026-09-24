<script lang="ts">
  import { onMount, tick } from "svelte";
  import TopBar from "./lib/components/TopBar.svelte";
  import Hero from "./lib/components/Hero.svelte";
  import Stage from "./lib/components/Stage.svelte";
  import SidePanel from "./lib/components/SidePanel.svelte";
  import SettingsSheet from "./lib/components/SettingsSheet.svelte";
  import DebugHud from "./lib/components/DebugHud.svelte";
  import { app } from "./lib/state.svelte";
  import { EngineClient, engineUrl } from "./lib/engine";
  import { VideoClock } from "./lib/clock";
  import { SyncEngine } from "./lib/sync";
  import { createDemoPlayer, createYouTubePlayer, type PlayerState, type VideoPlayer } from "./lib/player";
  import type { EngineMessage } from "./lib/types";

  // §6.7 buffering hysteresis (seconds of Telugu audio ready ahead of the playhead)
  const START_LEAD = 8, LOW_LEAD = 2, RESUME_LEAD = 12;

  let client: EngineClient | null = null;
  let player: VideoPlayer | null = null;
  let sync: SyncEngine | null = null;
  const clock = new VideoClock();
  let host = $state<HTMLDivElement>();
  let canvas = $state<HTMLCanvasElement>();
  let urlInput = $state<HTMLInputElement>();
  let wantPlay = false;
  let engineMissing = $state(false);
  let hudErrors = $state<number[]>([]);
  let hudLive = $state(0);
  let loadedVideo = "";

  const haveEnough = () =>
    app.readyUntil >= (app.duration || Infinity) - 0.5 ||
    (app.units.filter((u) => u.start >= app.time - 0.1).length >= 2 && app.lead >= START_LEAD);

  function onMessage(m: EngineMessage) {
    switch (m.type) {
      case "hello":
        app.backend = m.backend; app.device = m.device; app.demo = m.demo;
        break;
      case "video":
        app.video = { videoId: m.videoId, title: m.title, channel: m.channel, duration: m.duration, start: m.start };
        if (m.duration) app.duration = m.duration;
        if (m.videoId !== loadedVideo && (m.duration || !app.demo)) void mountPlayer(m.videoId, m.start, m.duration);
        break;
      case "status":
        app.stage = m.stage as typeof app.stage; app.message = m.message;
        break;
      case "unit": {
        const { type: _t, ...u } = m;
        app.units = [...app.units, u];
        sync?.addUnit(u);
        break;
      }
      case "unit_skipped":
        app.skipped = [...app.skipped, { id: m.id, start: m.start, end: m.end }];
        break;
      case "ready":
        app.readyUntil = m.until;
        break;
      case "speakers":
        app.speakers = m.speakers;
        break;
      case "stage_time":
        app.windowSeconds = [...app.windowSeconds.slice(-40), m.seconds];
        break;
      case "error":
        app.error = { message: m.message, retryable: m.retryable }; app.stage = "error";
        break;
    }
  }

  async function mountPlayer(videoId: string, start: number, duration: number) {
    loadedVideo = videoId;
    player?.destroy();
    sync?.dispose();
    await tick();
    const events = {
      ready: () => { app.duration = player?.duration() || app.duration; },
      state: onPlayerState,
      rate: (r: number) => clock.setRate(r),
      error: (msg: string) => { app.error = { message: msg, retryable: false }; },
    };
    try {
      player = app.demo && canvas
        ? createDemoPlayer(canvas, duration || 180, start, events)
        : host ? await createYouTubePlayer(host, videoId, start, events) : null;
    } catch (e) {
      app.error = { message: (e as Error).message, retryable: true };
      return;
    }
    if (!player) return;
    clock.anchor(start);
    app.time = start;
    sync = new SyncEngine(clock, player);
    sync.setVolume(app.volume);
    sync.speed = app.speed;
  }

  function onPlayerState(s: PlayerState) {
    if (!player || !sync) return;
    const t = player.time();
    if (s === "playing") {
      clock.setPlaying(true, t);
      app.playing = true;
      sync.start();
    } else if (s === "buffering") {
      clock.setPlaying(false, t);
    } else if (s === "paused" || s === "ended") {
      clock.setPlaying(false, t);
      if (sync.freezing) return; // a planned freeze-frame: the dub keeps playing
      app.playing = app.waitingForDub; // "waiting" still reads as playing intent
      sync.halt();
      if (s === "ended") { app.playing = false; wantPlay = false; }
    }
  }

  function play() {
    wantPlay = true;
    if (!player) return;
    if (haveEnough()) {
      app.waitingForDub = false;
      player.play();
    } else {
      app.waitingForDub = true;
      app.playing = true;
    }
  }

  function pause() {
    wantPlay = false;
    app.waitingForDub = false;
    app.playing = false;
    player?.pause();
    sync?.halt();
    app.remember();
  }

  const toggle = () => (app.playing ? pause() : play());

  function seek(t: number) {
    if (!player) return;
    t = Math.max(0, Math.min(t, app.duration || t));
    sync?.seek();
    player.seek(t);
    clock.anchor(t);
    app.time = t;
    client?.send({ type: "seek", time: t });
    if (wantPlay) {
      if (!haveEnough()) { player.pause(); app.waitingForDub = true; }
      else if (clock.playing) sync?.start();
    }
  }

  function setSpeed(s: number) {
    app.speed = s;
    if (sync) { sync.halt(); sync.speed = s; }
    player?.setRate(s);
    clock.setRate(s);
    client?.send({ type: "speed", speed: s });
    if (clock.playing) sync?.start();
  }

  function setVolume(v: number) {
    app.volume = v;
    sync?.setVolume(v);
  }

  function open(url: string) {
    app.resetVideo();
    wantPlay = false;
    loadedVideo = "";
    player?.destroy(); player = null;
    sync?.dispose(); sync = null;
    app.stage = "resolving";
    app.message = "Finding the video…";
    client?.send({ type: "open", url });
  }

  function setPreset(speaker: string, usePreset: boolean) {
    client?.send({ type: "speaker_preset", speaker, usePreset });
  }

  // Theme
  $effect(() => {
    const th = app.settings.theme;
    const dark = th === "dark" || (th === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  });

  onMount(() => {
    const url = engineUrl();
    if (!url) {
      engineMissing = true;
    } else {
      client = new EngineClient(url, {
        message: onMessage,
        audio: (a) => sync?.addAudio(a.id, a.sampleRate, a.samples),
        connection: (c) => (app.connection = c),
      });
      client.connect();
    }

    let raf = 0, lastPoll = 0, lastReport = 0;
    const loop = (ms: number) => {
      if (player) {
        if (clock.playing && ms - lastPoll > 250) { clock.observe(player.time(), ms); lastPoll = ms; }
        app.time = clock.now(ms);
        // §6.7 hysteresis: pause when the dub runs dry, resume with a comfortable lead
        if (wantPlay && app.waitingForDub && haveEnough() && app.lead >= RESUME_LEAD * (app.readyUntil >= app.duration - 1 ? 0 : 1)) {
          app.waitingForDub = false; player.play();
        } else if (wantPlay && clock.playing && !sync?.freezing && app.lead < LOW_LEAD && app.readyUntil < app.duration - 1 && !app.current) {
          app.waitingForDub = true; player.pause();
        }
        if (ms - lastReport > 1000) { client?.send({ type: "playhead", time: app.time }); lastReport = ms; }
        if (sync && app.settings.hud) { hudErrors = [...sync.stats.errorsMs]; hudLive = sync.scheduledCount; }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const autosave = setInterval(() => app.playing && app.remember(), 5000);

    const q = new URLSearchParams(location.search).get("v");
    if (q) setTimeout(() => open(q), 300);

    return () => { cancelAnimationFrame(raf); clearInterval(autosave); client?.close(); player?.destroy(); sync?.dispose(); };
  });

  function onKey(e: KeyboardEvent) {
    const typing = (e.target as HTMLElement)?.tagName === "INPUT";
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "l") { e.preventDefault(); urlInput?.focus(); urlInput?.select(); return; }
    if (typing || !app.video) return;
    if (e.key === " ") { e.preventDefault(); toggle(); }
    else if (e.key === "ArrowRight" && (e.target as HTMLElement)?.getAttribute("role") !== "slider") seek(app.time + 5);
    else if (e.key === "ArrowLeft" && (e.target as HTMLElement)?.getAttribute("role") !== "slider") seek(app.time - 5);
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="backdrop" aria-hidden="true"><div class="b1"></div><div class="b2"></div><div class="b3"></div><div class="grain"></div></div>

<div class="shell">
  <TopBar onOpen={open} bind:inputEl={urlInput} />

  <main>
    {#if engineMissing}
      <div class="notice">
        <h2>The Maata engine isn't running</h2>
        <p>Start it with <code>uv run maata-engine --ui app/dist</code> and open the address it prints, or launch the Maata app.</p>
      </div>
    {:else if !app.video}
      <Hero onOpen={open} />
      {#if app.error}<div class="toast" role="alert">{app.error.message}</div>{/if}
    {:else}
      <div class="watch">
        <Stage bind:host bind:canvas onToggle={toggle} onSeek={seek} onSpeed={setSpeed} onVolume={setVolume} />
        <SidePanel onPreset={setPreset} />
      </div>
    {/if}
  </main>
</div>

{#if app.settingsOpen}<SettingsSheet />{/if}
{#if app.settings.hud && app.video}<DebugHud errors={hudErrors} scheduled={hudLive} />{/if}

<style>
  .shell { position: relative; z-index: 1; height: 100%; display: flex; flex-direction: column; }
  main { flex: 1; overflow: auto; }
  .watch {
    display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 22px; align-items: start;
    max-width: 1500px; margin: 0 auto; padding: 10px 28px 32px;
  }
  @media (max-width: 1080px) { .watch { grid-template-columns: 1fr; } }

  .backdrop { position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; }
  .backdrop > div:not(.grain) { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.5; }
  .b1 { width: 55vw; height: 55vw; left: -18vw; top: -26vw; background: radial-gradient(circle, rgb(247 183 51 / 0.45), transparent 65%); animation: drift 28s ease-in-out infinite alternate; }
  .b2 { width: 50vw; height: 50vw; right: -20vw; top: -10vw; background: radial-gradient(circle, rgb(216 51 107 / 0.4), transparent 65%); animation: drift 34s ease-in-out infinite alternate-reverse; }
  .b3 { width: 60vw; height: 60vw; left: 20vw; bottom: -40vw; background: radial-gradient(circle, rgb(90 60 200 / 0.35), transparent 65%); animation: drift 40s ease-in-out infinite alternate; }
  :global([data-theme="light"]) .backdrop > div:not(.grain) { opacity: 0.3; }
  .grain { position: absolute; inset: 0; opacity: 0.05; background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>"); }
  @keyframes drift { to { transform: translate(6vw, 4vw) scale(1.08); } }

  .notice { max-width: 560px; margin: 18vh auto; text-align: center; padding: 28px; border-radius: var(--r-xl); background: var(--surface); border: 1px solid var(--border); }
  .notice h2 { margin: 0 0 8px; }
  .notice p { color: var(--text-2); margin: 0; }
  code { font: 12px ui-monospace, monospace; background: var(--surface-strong); padding: 2px 6px; border-radius: 6px; }
  .toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); padding: 10px 16px; border-radius: 12px; background: rgb(255 93 93 / 0.14); border: 1px solid rgb(255 93 93 / 0.4); color: #ffd2d2; }
</style>
