<script lang="ts">
  import { onMount, tick } from "svelte";
  import TopBar from "./lib/components/TopBar.svelte";
  import Hero from "./lib/components/Hero.svelte";
  import Stage from "./lib/components/Stage.svelte";
  import SidePanel from "./lib/components/SidePanel.svelte";
  import SettingsSheet from "./lib/components/SettingsSheet.svelte";
  import DebugHud from "./lib/components/DebugHud.svelte";
  import ClaudeBanner from "./lib/components/ClaudeBanner.svelte";
  import { app } from "./lib/state.svelte";
  import { EngineClient, engineUrl } from "./lib/engine";
  import { VideoClock } from "./lib/clock";
  import { SyncEngine, unitSpan, type VideoControl } from "./lib/sync";
  import { ASK_AHEAD, MissingAudio, PLAY_NOW_MIN, bufferStep, startDecision } from "./lib/buffer";
  import { createDemoPlayer, createYouTubePlayer, type PlayerEvents, type PlayerState, type VideoPlayer } from "./lib/player";
  import type { ClientMessage, EngineMessage } from "./lib/types";

  // Webview memory (§6.8, ARCHITECTURE §3.12): keep dub audio from 1 min behind the playhead to the engine's look-ahead
  // ahead of it, the window the engine keeps in memory and sends. The rest waits on the engine's disk until asked for.
  const KEEP_BEHIND = 60;
  // A seek makes the engine re-send all the dub it has near the new playhead, up to the look-ahead: a burst
  // of seeks (a scrubber drag, arrow-key repeats) goes to the engine once, when it settles.
  const SEEK_SETTLE_MS = 250;

  let client: EngineClient | null = null;
  const send = (m: ClientMessage) => client?.send(m);
  let player: VideoPlayer | null = null;
  const clock = new VideoClock();
  // One SyncEngine (one AudioContext) for the app's life: units and audio may arrive before the player is up.
  const video: VideoControl = {
    pause: () => player?.pause(),
    play: () => player?.play(),
    setRate: (r) => player?.setRate(r),
    rates: () => player?.rates() ?? [1],
  };
  let sync: SyncEngine | null = null;
  const ensureSync = () => (sync ??= new SyncEngine(clock, video));
  let host = $state<HTMLDivElement>();
  let canvas = $state<HTMLCanvasElement>();
  let urlInput = $state<HTMLInputElement>();
  let wantPlay = false;
  let engineMissing = $state(false);
  let hudErrors = $state<number[]>([]);
  let hudLive = $state(0);
  // The video plays but the system keeps the dub's audio output off (WebKit: no user gesture started it).
  let audioBlocked = $state(false);
  let loadedVideo = "";
  const seen = new Set<number>();
  /** Units whose audio was evicted, until the engine re-sends it. */
  const missing = new MissingAudio();
  let seekTimer = 0;

  function onMessage(m: EngineMessage) {
    switch (m.type) {
      case "hello":
        app.backend = m.backend; app.device = m.device; app.demo = m.demo;
        app.setClaude(m.claude);
        break;
      case "claude_error": {
        // Translation waits and says why; lines already translated keep playing (not a pipeline error).
        const { type: _t, ...problem } = m;
        app.claudeError(problem);
        break;
      }
      case "claude_ok":
        app.claudeOk();
        break;
      case "video":
        app.video = { videoId: m.videoId, title: m.title, channel: m.channel, duration: m.duration, start: m.start };
        if (m.duration) app.duration = m.duration;
        if (m.videoId !== loadedVideo && (m.duration || !app.demo)) void mountPlayer(m.videoId, m.start, m.duration);
        break;
      case "status":
        app.setStatus(m.stage, m.message);
        break;
      case "unit": {
        const { type: _t, ...u } = m;
        const i = app.units.findIndex((x) => x.id === u.id);
        app.units = i < 0 ? [...app.units, u] : app.units.map((x, k) => (k === i ? u : x));
        if (!seen.has(u.id)) { seen.add(u.id); app.lineCount = seen.size; }
        sync?.addUnit(u);
        if (u.audio === false) {
          // Beyond the engine's window: its audio is asked for as the playhead nears (reaskMissing).
          sync?.dropAudio(u.id);
          missing.add([unitSpan(u)]);
        }
        break;
      }
      case "unit_skipped":
        if (!app.skipped.some((s) => s.id === m.id)) app.skipped = [...app.skipped, { id: m.id, start: m.start, end: m.end }];
        break;
      case "ready":
        app.readyUntil = m.until;
        app.readyRanges = m.ranges ?? [];
        app.throughput = m.throughput ?? 0;
        app.targetLead = m.targetLead ?? 0;
        break;
      case "speaker_scan": {
        const { type: _t, ...scan } = m;
        app.speakerScan = scan;
        break;
      }
      case "speakers":
        app.speakers = m.speakers;
        break;
      case "stage_time":
        app.windowSeconds = [...app.windowSeconds.slice(-40), m.seconds];
        break;
      case "notice":
        app.addNotice({ kind: m.kind, from: m.from, to: m.to });
        break;
      case "error":
        app.error = { message: m.message, retryable: m.retryable }; app.stage = "error";
        break;
    }
  }

  async function mountPlayer(videoId: string, start: number, duration: number) {
    loadedVideo = videoId;
    player?.destroy();
    player = null;
    sync?.halt();
    await tick();
    const events: PlayerEvents = {
      ready: (info) => {
        if (loadedVideo !== videoId) return;
        if (info.duration) app.duration = info.duration;
        client?.send({ type: "player", rates: info.rates });
      },
      state: onPlayerState,
      rate: (r) => clock.setRate(r),
      error: (msg) => { app.error = { message: msg, retryable: false }; },
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
    clock.setPlaying(false, start);
    app.time = start;
    const s = ensureSync();
    s.setVolume(app.volume);
    s.speed = app.speed;
  }

  function onPlayerState(s: PlayerState) {
    if (!player || !sync) return;
    const t = player.time();
    if (s === "playing") {
      clock.setPlaying(true, t);
      app.playing = true;
      app.started = true;
      app.ranDry = false;
      sync.start();
    } else if (s === "buffering") {
      // The video stalled: stop the dub with it (resuming where each line was cut) so they never drift apart.
      clock.setPlaying(false, t);
      if (!sync.freezing) sync.halt();
    } else if (s === "paused" || s === "ended") {
      clock.setPlaying(false, t);
      if (sync.freezing) return; // a planned freeze-frame: the dub keeps playing
      app.playing = app.waitingForDub; // "waiting" still reads as playing intent
      sync.halt();
      if (s === "ended") { app.playing = false; app.waitingForDub = false; wantPlay = false; }
    }
  }

  function play() {
    wantPlay = true;
    sync?.unlock(); // inside the click: the video may start later, once enough dub is banked (see SyncEngine.unlock)
    if (!player) return;
    if (!app.mustWait) {
      app.waitingForDub = false;
      player.play();
    } else {
      app.waitingForDub = true;
      app.playing = true;
    }
  }

  /** Start before the full lead is banked; if the dub runs out, playback waits for it again. */
  function playNow() {
    if (!player || app.lead < PLAY_NOW_MIN) return;
    sync?.unlock();
    wantPlay = true;
    app.waitingForDub = false;
    app.playing = true;
    player.play();
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
    const held = sync?.seek() ?? false;
    player.seek(t);
    clock.anchor(t);
    app.time = t;
    tellSeek(t);
    missing.forgetBefore(t); // before deciding: a line the engine won't re-send mustn't hold the lead at 0
    if (missing.dirty) app.missing = missing.ranges();
    if (wantPlay && !app.waitingForDub) {
      if (startDecision(app.lead, app.need, false) === "wait") { app.waitingForDub = true; player.pause(); }
      else sync?.carryOn(held); // rolls the video too if the seek cut a freeze's hold short
    }
  }

  /** Tell the engine the playhead moved (it re-sends the dub it has near t), once a burst of seeks settles. */
  function tellSeek(t: number) {
    clearTimeout(seekTimer);
    seekTimer = window.setTimeout(() => {
      client?.send({ type: "seek", time: t });
      missing.asked(t, app.lookahead, performance.now());
    }, SEEK_SETTLE_MS);
  }

  function setSpeed(s: number) {
    app.speed = s;
    const held = sync?.halt() ?? false;
    if (sync) sync.speed = s;
    player?.setRate(s);
    clock.setRate(s);
    client?.send({ type: "speed", speed: s });
    // A paused YouTube player stays paused through a rate change: if this cut a freeze's hold short, carry on.
    if (clock.playing || (wantPlay && !app.waitingForDub)) sync?.carryOn(held);
  }

  function setVolume(v: number) {
    app.volume = v;
    sync?.setVolume(v);
  }

  function open(url: string) {
    wantPlay = false;
    loadedVideo = "";
    player?.destroy(); player = null;
    ensureSync().reset();
    clock.setPlaying(false, 0);
    seen.clear();
    missing.clear();
    clearTimeout(seekTimer);
    app.openVideo(url, send); // resets the video state and sends `open` with the settings (lib/state.svelte.ts)
  }

  /** The side panel confirms a switch to the stock voice before this runs; the engine's `speakers` reply is the truth. */
  const setPreset = (speaker: string, usePreset: boolean) => app.setPreset(speaker, usePreset, send);

  /** Drop dub audio far from the playhead; remember what went so a return can wait for it. */
  function evict() {
    if (sync) missing.add(sync.evict(app.time - KEEP_BEHIND, app.time + app.lookahead));
  }

  /** Audio the UI doesn't hold that the playhead is nearing: ask the engine for it by id (lib/buffer.ts MissingAudio). */
  function reaskMissing(ms: number) {
    const ids = missing.ask(app.time, app.time + ASK_AHEAD, ms);
    if (ids.length) send({ type: "audio", ids });
  }

  /** "Prepare the whole video": the engine dubs to the end; memory stays windowed on both sides. */
  function setPrepareAll(on: boolean) {
    app.prepareAll = on;
    send({ type: "prepare", whole: on });
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
        audio: (a) => {
          sync?.addAudio(a.id, a.sampleRate, a.samples);
          missing.got(a.id);
        },
        connection: (c) => (app.connection = c),
      });
      client.connect();
    }

    let raf = 0, lastPoll = 0, lastReport = 0, lastEvict = 0, lastAsk = 0, lastDiag = 0;
    const loop = (ms: number) => {
      if (missing.dirty) app.missing = missing.ranges();
      if (player) {
        if (clock.playing && ms - lastPoll > 250) { clock.observe(player.time(), ms); lastPoll = ms; }
        app.time = clock.now(ms);
        // Buffer policy (lib/buffer.ts): bank `need` before starting, stall only when the dub runs out.
        const step = bufferStep({
          wantPlay, waiting: app.waitingForDub, playing: clock.playing, freezing: !!sync?.freezing,
          linePlaying: app.current !== null, lead: app.lead, need: app.need, remaining: app.remaining,
        });
        if (step === "start") {
          app.waitingForDub = false; player.play();
        } else if (step === "stall") {
          app.waitingForDub = true; app.ranDry = true; player.pause();
        }
        if (ms - lastReport > 1000) {
          // Not playing (paused, or waiting for the dub): a slow engine uses the time to work further ahead.
          client?.send({ type: "playhead", time: app.time, playing: wantPlay && !app.waitingForDub });
          lastReport = ms;
        }
        if (ms - lastEvict > 5000) { evict(); lastEvict = ms; }
        if (missing.size && ms - lastAsk > 1000) { reaskMissing(ms); lastAsk = ms; }
        if (sync && app.settings.hud) { hudErrors = [...sync.stats.errorsMs]; hudLive = sync.scheduledCount; }
        if (sync) audioBlocked = clock.playing && sync.audioState !== "running";
        if (sync && wantPlay && ms - lastDiag > 5000) {
          client?.send({
            type: "diag", ...sync.diag(), volume: app.volume, lead: Math.round(app.lead * 10) / 10,
            waiting: app.waitingForDub, missing: missing.size, time: Math.round(app.time * 10) / 10,
          });
          lastDiag = ms;
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const autosave = setInterval(() => app.playing && app.remember(), 5000);

    const q = new URLSearchParams(location.search).get("v");
    if (q) setTimeout(() => open(q), 300);

    return () => {
      cancelAnimationFrame(raf); clearInterval(autosave); clearTimeout(seekTimer); client?.close(); player?.destroy(); sync?.dispose();
    };
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

<svelte:window onkeydown={onKey} onpointerdown={() => sync?.unlock()} />

<div class="backdrop" aria-hidden="true"><div class="b1"></div><div class="b2"></div><div class="b3"></div><div class="grain"></div></div>

<div class="shell">
  <TopBar onOpen={open} bind:inputEl={urlInput} />

  <main>
    <ClaudeBanner />
    {#if engineMissing}
      <div class="notice">
        <h2>The Maata engine isn't running</h2>
        <p>Start it with <code>uv run maata-engine --ui app/dist</code> and open the address it prints, or launch the Maata app.</p>
      </div>
    {:else if !app.video}
      <Hero onOpen={open} />
      {#if app.error}<div class="toast" role="alert">{app.error.message}</div>{/if}
    {:else}
      {#if audioBlocked}
        <button class="toast audio-off" onclick={() => sync?.unlock()}>
          macOS is keeping the Telugu audio off. Click here to turn it on.
        </button>
      {/if}
      <div class="watch">
        <Stage bind:host bind:canvas onToggle={toggle} onPlayNow={playNow} onSeek={seek} onSpeed={setSpeed} onVolume={setVolume} />
        <SidePanel onPreset={setPreset} onPrepareAll={setPrepareAll} />
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
  .audio-off { z-index: 10; font: inherit; cursor: pointer; }
</style>
