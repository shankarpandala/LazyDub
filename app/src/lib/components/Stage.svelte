<script lang="ts">
  import Icon from "./Icon.svelte";
  import Scrubber from "./Scrubber.svelte";
  import { app } from "../state.svelte";
  import { fmtTime } from "../format";

  let {
    host = $bindable(), canvas = $bindable(), onToggle, onSeek, onSpeed, onVolume,
  }: {
    host?: HTMLDivElement; canvas?: HTMLCanvasElement;
    onToggle: () => void; onSeek: (t: number) => void; onSpeed: (s: number) => void; onVolume: (v: number) => void;
  } = $props();

  const speeds = [0.75, 1, 1.25, 1.5];
  let speedOpen = $state(false);
  let lastVolume = 1;

  const chip = $derived.by(() => {
    if (app.error) return { tone: "danger", text: app.error.message };
    if (app.waitingForDub) return { tone: "warn", text: `Preparing Telugu audio… ${Math.round(app.lead)} s` };
    if (app.stage === "resolving" || app.stage === "fetching") return { tone: "warn", text: app.message };
    if (!app.units.length) return { tone: "warn", text: "Preparing Telugu audio…" };
    if (app.readyUntil >= (app.duration || Infinity) - 1) return { tone: "ok", text: "Telugu audio ready" };
    return { tone: "ok", text: `Telugu ready ${Math.round(app.lead)} s ahead` };
  });
</script>

<div class="stage">
  <div class="frame">
    {#if app.demo}
      <canvas class="video" bind:this={canvas}></canvas>
    {:else}
      <div class="video" bind:this={host}></div>
    {/if}
    <button class="shield" aria-label={app.playing ? "Pause" : "Play"} onclick={onToggle}></button>

    <div class="badge" title="The Telugu voices are AI-generated">
      <Icon name="sparkle" size={13} />
      <span>AI dub</span>
      <span class="te sep">తెలుగు</span>
    </div>
    <div class={`chip ${chip.tone}`} role="status" aria-live="polite">
      <span class="pulse"></span>{chip.text}
    </div>

    {#if !app.playing}
      <button class="bigplay" aria-label="Play" onclick={onToggle}><Icon name="play" size={30} /></button>
    {/if}
  </div>

  <div class="title-row">
    <div class="title">
      <div class="t">{app.video?.title || "Loading video…"}</div>
      {#if app.video?.channel}<div class="c">{app.video.channel}</div>{/if}
    </div>
  </div>

  <div class="controls">
    <Scrubber {onSeek} />
    <div class="row">
      <div class="left">
        <button class="ctl" aria-label="Back 5 seconds" onclick={() => onSeek(Math.max(0, app.time - 5))}><Icon name="back5" /></button>
        <button class="ctl play" aria-label={app.playing ? "Pause" : "Play"} onclick={onToggle}>
          <Icon name={app.playing ? "pause" : "play"} size={20} />
        </button>
        <button class="ctl" aria-label="Forward 5 seconds" onclick={() => onSeek(app.time + 5)}><Icon name="fwd5" /></button>
        <div class="time tabular">{fmtTime(app.time)} <span>/ {fmtTime(app.duration || app.video?.duration || 0)}</span></div>
      </div>
      <div class="right">
        <div class="vol">
          <button class="ctl" aria-label={app.volume ? "Mute dub" : "Unmute dub"} onclick={() => { if (app.volume) { lastVolume = app.volume; onVolume(0); } else onVolume(lastVolume || 1); }}>
            <Icon name={app.volume ? "volume" : "mute"} />
          </button>
          <input type="range" min="0" max="1" step="0.01" value={app.volume} aria-label="Dub volume" oninput={(e) => onVolume(+(e.currentTarget as HTMLInputElement).value)} />
        </div>
        <div class="speed">
          <button class="pill tabular" aria-haspopup="menu" aria-expanded={speedOpen} onclick={() => (speedOpen = !speedOpen)}>{app.speed}×</button>
          {#if speedOpen}
            <div class="menu" role="menu">
              {#each speeds as s (s)}
                <button role="menuitemradio" aria-checked={s === app.speed} class:on={s === app.speed} onclick={() => { onSpeed(s); speedOpen = false; }}>{s}×</button>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .stage { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .frame {
    position: relative; aspect-ratio: 16 / 9; width: 100%;
    border-radius: var(--r-xl); overflow: hidden; background: #000;
    box-shadow: var(--shadow-lg), 0 40px 120px -40px rgb(242 96 60 / 0.35);
  }
  .video { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .video :global(iframe) { width: 100%; height: 100%; border: 0; pointer-events: none; }
  .shield { position: absolute; inset: 0; background: transparent; border: 0; cursor: pointer; }
  .badge, .chip {
    position: absolute; top: 14px; display: flex; align-items: center; gap: 6px;
    height: 28px; padding: 0 11px; border-radius: 999px; font-size: 12px; font-weight: 600;
    background: rgb(10 9 16 / 0.62); backdrop-filter: var(--blur); border: 1px solid rgb(255 255 255 / 0.12); color: #fff;
    pointer-events: none;
  }
  .badge { left: 14px; }
  .badge :global(svg) { color: var(--turmeric); }
  .badge .sep { font-weight: 500; opacity: 0.75; padding-left: 6px; border-left: 1px solid rgb(255 255 255 / 0.2); }
  .chip { right: 14px; font-weight: 500; max-width: 60%; }
  .pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); flex: none; }
  .chip.warn .pulse { background: var(--warn); animation: blink 1.2s ease-in-out infinite; }
  .chip.danger .pulse { background: var(--danger); }
  .bigplay {
    position: absolute; left: 50%; top: 50%; width: 76px; height: 76px; margin: -38px 0 0 -38px; border-radius: 50%;
    border: 0; cursor: pointer; display: grid; place-items: center; padding-left: 5px;
    background: var(--accent-grad); color: #1a0f08; box-shadow: var(--accent-glow), 0 20px 60px rgb(0 0 0 / 0.5);
    transition: transform 0.2s var(--ease);
  }
  .bigplay:hover { transform: scale(1.06); }
  .title-row { display: flex; justify-content: space-between; align-items: flex-start; padding: 0 4px; }
  .title .t { font-size: 17px; font-weight: 620; letter-spacing: -0.01em; }
  .title .c { color: var(--text-2); font-size: 13px; margin-top: 2px; }
  .controls {
    padding: 8px 14px 10px; border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur);
  }
  .row { display: flex; justify-content: space-between; align-items: center; }
  .left, .right { display: flex; align-items: center; gap: 4px; }
  .ctl {
    width: 36px; height: 36px; border-radius: 10px; border: 0; background: transparent; color: var(--text-2);
    display: grid; place-items: center; cursor: pointer;
  }
  .ctl:hover { background: var(--surface-hover); color: var(--text); }
  .ctl.play { width: 42px; height: 42px; border-radius: 50%; color: var(--text); background: var(--surface-strong); }
  .time { margin-left: 10px; font-size: 13px; font-weight: 560; }
  .time span { color: var(--text-3); font-weight: 500; }
  .vol { display: flex; align-items: center; gap: 2px; }
  input[type="range"] { width: 90px; accent-color: var(--vermilion); }
  .speed { position: relative; }
  .pill {
    height: 30px; min-width: 50px; padding: 0 10px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-weight: 600; font-size: 12px; cursor: pointer;
  }
  .menu {
    position: absolute; right: 0; bottom: 38px; display: flex; flex-direction: column; padding: 6px; gap: 2px;
    background: var(--bg-raised); border: 1px solid var(--border-strong); border-radius: var(--r-md); box-shadow: var(--shadow-lg); z-index: 10;
  }
  .menu button { border: 0; background: transparent; padding: 6px 16px; border-radius: 8px; text-align: left; cursor: pointer; font-weight: 560; }
  .menu button:hover { background: var(--surface-hover); }
  .menu button.on { color: var(--vermilion); }
  @keyframes blink { 50% { opacity: 0.35; } }
</style>
