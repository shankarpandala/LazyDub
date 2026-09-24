<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import { speakerHue } from "../format";

  let { onPreset }: { onPreset: (speaker: string, usePreset: boolean) => void } = $props();

  const stages = [
    { key: "fetch", label: "Fetch audio", te: "ఆడియో" },
    { key: "listen", label: "Listen", te: "వినడం" },
    { key: "translate", label: "Translate", te: "అనువాదం" },
    { key: "voice", label: "Voice", te: "స్వరం" },
  ];
  const progress = $derived.by(() => {
    const order = ["idle", "resolving", "fetching", "listening", "translating", "voicing", "ready"];
    const allReady = app.units.length > 0 && app.readyUntil >= (app.duration || Infinity) - 1;
    const i = allReady ? 6 : app.units.length ? 5 : Math.max(order.indexOf(app.stage), 0);
    return stages.map((_, k) => (i > k + 2 ? "done" : i === k + 2 ? "active" : "todo"));
  });
  const leadPct = $derived(Math.min(100, (app.lead / 30) * 100));
  const clonedCount = $derived(app.speakers.filter((s) => s.voice === "cloned" && !s.usePreset).length);
</script>

<aside class="side">
  <section class="card">
    <header>
      <h2>Pipeline</h2>
      <span class="muted tabular">{app.units.length} lines</span>
    </header>
    <ol class="stages">
      {#each stages as s, i (s.key)}
        <li class={progress[i]}>
          <span class="bullet">{#if progress[i] === "done"}<Icon name="check" size={12} />{/if}</span>
          <span class="lbl">{s.label}</span>
          <span class="te muted">{s.te}</span>
        </li>
      {/each}
    </ol>
    <div class="lead">
      <div class="lead-row"><span>Telugu buffered ahead</span><span class="tabular">{Math.round(app.lead)} s</span></div>
      <div class="meter"><div style={`width:${leadPct}%`}></div></div>
    </div>
  </section>

  <section class="card">
    <header>
      <h2>Speakers</h2>
      <span class="muted">{clonedCount}/{app.speakers.length} cloned</span>
    </header>
    {#if !app.speakers.length}
      <p class="empty">Speakers appear as they're detected.</p>
    {/if}
    <ul class="speakers">
      {#each app.speakers as s (s.id)}
        {@const hue = speakerHue(s.id)}
        {@const pct = Math.min(1, s.referenceSeconds / 6)}
        <li>
          <div class="avatar" style={`--h:${hue}`}>
            <svg viewBox="0 0 36 36" aria-hidden="true">
              <circle cx="18" cy="18" r="16" class="ring-bg" />
              <circle cx="18" cy="18" r="16" class="ring" style={`stroke-dasharray:${pct * 100.5} 100.5`} />
            </svg>
            <span>{s.label.replace("Speaker ", "")}</span>
          </div>
          <div class="info">
            <div class="name">{s.label}</div>
            <div class="voice">
              {#if s.usePreset}Preset voice
              {:else if s.voice === "cloned"}<span class="cloned">Own voice · cloned</span>
              {:else}Preset voice · learning {Math.round(s.referenceSeconds)}/6 s{/if}
            </div>
          </div>
          <label class="toggle" title="Use a preset voice for this speaker">
            <input type="checkbox" checked={s.usePreset} onchange={(e) => onPreset(s.id, (e.currentTarget as HTMLInputElement).checked)} />
            <span class="track"><span class="thumb"></span></span>
            <span class="sr-only">Use preset voice for {s.label}</span>
          </label>
        </li>
      {/each}
    </ul>
  </section>

  {#if app.current}
    <section class="card now">
      <header><h2>Now speaking</h2><span class="muted">{app.current.voice === "cloned" ? "cloned voice" : "preset voice"}</span></header>
      <p class="te line">{app.current.telugu}</p>
    </section>
  {/if}

  <p class="note"><Icon name="sparkle" size={13} /> Voices are AI-generated from the original speakers, for private viewing.</p>
</aside>

<style>
  .side { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .card { padding: 16px; border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur); }
  header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
  h2 { margin: 0; font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.09em; color: var(--text-2); }
  .muted { color: var(--text-3); font-size: 12px; }
  .stages { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 9px; }
  .stages li { display: grid; grid-template-columns: 20px 1fr auto; align-items: center; gap: 8px; color: var(--text-3); font-weight: 520; }
  .bullet { width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid var(--border-strong); display: grid; place-items: center; }
  li.done { color: var(--text); }
  li.done .bullet { background: var(--ok); border-color: var(--ok); color: #06150e; }
  li.active { color: var(--text); }
  li.active .bullet { border-color: var(--turmeric); box-shadow: 0 0 0 4px rgb(247 183 51 / 0.15); animation: pulse 1.4s ease-in-out infinite; }
  .lead { margin-top: 16px; }
  .lead-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-2); margin-bottom: 6px; }
  .meter { height: 6px; border-radius: 99px; background: rgb(255 255 255 / 0.08); overflow: hidden; }
  .meter div { height: 100%; background: var(--accent-grad); border-radius: 99px; transition: width 0.4s var(--ease); }
  .empty { margin: 0; color: var(--text-3); font-size: 13px; }
  .speakers { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
  .speakers li { display: grid; grid-template-columns: 40px 1fr auto; align-items: center; gap: 12px; }
  .avatar { position: relative; width: 40px; height: 40px; display: grid; place-items: center; font-weight: 700; color: hsl(var(--h) 90% 72%); }
  .avatar span { position: relative; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; background: hsl(var(--h) 70% 50% / 0.16); font-size: 13px; }
  .avatar svg { position: absolute; inset: 0; transform: rotate(-90deg); }
  .ring-bg { fill: none; stroke: rgb(255 255 255 / 0.08); stroke-width: 2.5; }
  .ring { fill: none; stroke: hsl(var(--h) 85% 62%); stroke-width: 2.5; stroke-linecap: round; transition: stroke-dasharray 0.5s var(--ease); }
  .name { font-weight: 600; }
  .voice { font-size: 12px; color: var(--text-3); }
  .cloned { color: var(--ok); }
  .toggle { cursor: pointer; }
  .toggle input { position: absolute; opacity: 0; }
  .track { display: block; width: 34px; height: 20px; border-radius: 99px; background: var(--surface-strong); border: 1px solid var(--border); position: relative; transition: background 0.2s; }
  .thumb { position: absolute; top: 2px; left: 2px; width: 14px; height: 14px; border-radius: 50%; background: var(--text-2); transition: transform 0.2s var(--ease), background 0.2s; }
  .toggle input:checked + .track { background: rgb(242 96 60 / 0.35); }
  .toggle input:checked + .track .thumb { transform: translateX(14px); background: #fff; }
  .toggle input:focus-visible + .track { outline: 2px solid var(--turmeric); outline-offset: 2px; }
  .now .line { margin: 0; font-size: 17px; line-height: 1.6; }
  .note { display: flex; gap: 6px; align-items: center; margin: 0 4px; font-size: 12px; color: var(--text-3); }
  .note :global(svg) { color: var(--turmeric); flex: none; }
  @keyframes pulse { 50% { box-shadow: 0 0 0 7px rgb(247 183 51 / 0.05); } }
</style>
