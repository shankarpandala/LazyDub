<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import { fmtTime, speakerHue } from "../format";
  import {
    clonedLabel, cloneStrengthLabel, flipPreset, matchSummary, preparedAhead, presetConfirmBody, settlePreset, speakerStatus, voiceNote,
    type PresetStep,
  } from "../voice";
  import type { Speaker, SpeakerStatus } from "../types";

  let { onPreset, onPrepareAll }: {
    onPreset: (speaker: string, usePreset: boolean) => void; onPrepareAll: (on: boolean) => void;
  } = $props();

  const stages = [
    { key: "fetch", label: "Fetch audio", te: "ఆడియో" },
    { key: "listen", label: "Listen", te: "వినడం" },
    { key: "translate", label: "Translate", te: "అనువాదం" },
    { key: "voice", label: "Voice", te: "స్వరం" },
  ];
  const progress = $derived.by(() => {
    const order = ["idle", "resolving", "fetching", "listening", "translating", "voicing", "ready"];
    const i = app.units.length && app.allReady ? 6 : app.units.length ? 5 : Math.max(order.indexOf(app.stage), 0);
    return stages.map((_, k) => (i > k + 2 ? "done" : i === k + 2 ? "active" : "todo"));
  });
  const leadPct = $derived(app.allReady ? 100 : Math.min(100, (app.lead / Math.max(app.need, 1)) * 100));

  const STATUS: Record<SpeakerStatus, string> = { found: "Found", cloning: "Cloning…", cloned: "Cloned", preset: "Preset voice" };
  const statusOf = speakerStatus;
  const totalTalk = $derived(app.speakers.reduce((a, s) => a + (s.talkSeconds || 0), 0));

  const clonedCount = $derived(app.speakers.filter((s) => statusOf(s) === "cloned").length);
  const presetCount = $derived(app.speakers.filter((s) => statusOf(s) === "preset").length);
  const scan = $derived.by(() => {
    const sc = app.speakerScan;
    if (!sc) return null;
    if (sc.phase === "scanning") {
      // The engine sends video times; its pre-pass runs from where the video opened, so count from there.
      const from = app.video?.start ?? 0;
      const span = Math.max(0, sc.duration - from);
      const done = Math.min(span, Math.max(0, sc.until - from));
      return { text: `Finding speakers ${fmtTime(done)} / ${fmtTime(span)}`, pct: span ? done / span : 0, busy: true };
    }
    if (sc.phase === "cloning") {
      const n = app.speakers.filter((s) => statusOf(s) === "cloning" || statusOf(s) === "found").length || sc.found;
      return { text: `Cloning ${n} voice${n === 1 ? "" : "s"}`, pct: 1, busy: true };
    }
    if (presetCount) return { text: `${presetCount} on preset voice`, pct: 1, busy: false, warn: true };
    return { text: "Voices ready", pct: 1, busy: false };
  });
  /** The requested voice match, and whether the engine reports building the voices with it (never assumed). */
  const match = $derived(matchSummary(app.cloneStrength, app.speakers));

  // Switching a speaker to the stock voice is confirmed first; switching back to their clone is immediate (lib/voice.ts).
  let confirming = $state<string | null>(null);
  const switchId = (id: string) => `preset-switch-${id}`;

  function apply(step: PresetStep) {
    confirming = step.confirming;
    if (step.send) onPreset(step.send.speaker, step.send.usePreset);
  }

  const flip = (s: Speaker) => apply(flipPreset(s, confirming));

  function settle(id: string, usePreset: boolean) {
    apply(settlePreset(id, usePreset));
    document.getElementById(switchId(id))?.focus();
  }

  /** Space and Enter act on these controls, not on the page's play/pause shortcut. */
  function ownKeys(e: KeyboardEvent, id?: string) {
    if (e.key === " " || e.key === "Enter") e.stopPropagation();
    else if (e.key === "Escape" && id) { e.stopPropagation(); settle(id, false); }
  }

  const focusOnMount = (node: HTMLElement) => node.focus();
</script>

<aside class="side">
  <section class="card">
    <header>
      <h2>Pipeline</h2>
      <span class="muted tabular">{app.lineCount} lines</span>
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
      <div class="lead-row">
        <span>Telugu ready ahead</span>
        <span class="tabular">{app.allReady ? "to the end" : `${fmtTime(app.lead)} / ${fmtTime(app.need)}`}</span>
      </div>
      <div class="meter"><div style={`width:${leadPct}%`}></div></div>
      {#if app.throughput > 0}
        <div class="rate muted tabular">Dubbing at {app.throughput.toFixed(1)}× real time</div>
      {/if}
      {#if !app.allReady || app.prepareAll}
        <button
          class="btn whole"
          aria-pressed={app.prepareAll}
          title={app.prepareAll ? "Stop at the look-ahead again" : "Dub on to the end of the video, beyond the look-ahead"}
          onclick={() => onPrepareAll(!app.prepareAll)}
        >{app.prepareAll ? "Preparing the whole video" : "Prepare the whole video"}</button>
      {/if}
    </div>
  </section>

  <section class="card">
    <header>
      <h2>Speakers</h2>
      {#if scan}
        <span class="scan" class:busy={scan.busy} class:ok={!scan.busy && !scan.warn} class:warn={scan.warn}>
          {#if scan.warn}<Icon name="alert" size={12} />{:else if !scan.busy}<Icon name="check" size={12} />{/if}<span class="tabular">{scan.text}</span>
        </span>
      {:else if app.speakers.length}
        <span class="muted" class:warn-text={presetCount > 0}>{clonedCount}/{app.speakers.length} cloned{presetCount ? ` · ${presetCount} preset` : ""}</span>
      {/if}
    </header>
    <div class="match" title={match.title}>
      <span class="muted">Voice match</span>
      <span class="match-mode">{match.mode}</span>
      {#if match.note}<span class="match-note" class:warn-text={match.warn}>{match.note}</span>{/if}
    </div>
    {#if scan?.busy && app.speakerScan?.phase === "scanning"}
      <div class="meter scan-meter"><div style={`width:${Math.min(100, scan.pct * 100)}%`}></div></div>
    {/if}
    {#if !app.speakers.length}
      <p class="empty">Speakers appear as they're detected.</p>
    {/if}
    <ul class="speakers">
      {#each app.speakers as s (s.id)}
        {@const hue = speakerHue(s.id)}
        {@const st = statusOf(s)}
        {@const share = totalTalk ? (s.talkSeconds || 0) / totalTalk : 0}
        {@const note = voiceNote(s, app.units, app.time)}
        {@const mode = s.cloneStrength && s.cloneStrength !== app.cloneStrength ? `Uses ${cloneStrengthLabel(s.cloneStrength)}` : ""}
        <li class:preset={st === "preset"}>
          <div class="row">
            <div class={`avatar ${st}`} style={`--h:${hue}`}>
              <svg viewBox="0 0 36 36" aria-hidden="true">
                <circle cx="18" cy="18" r="16" class="ring-bg" />
                <circle cx="18" cy="18" r="16" class="ring" />
              </svg>
              <span>{s.label.replace("Speaker ", "")}</span>
            </div>
            <div class="info">
              <div class="name-row">
                <span class="name">{s.label}</span>
                <span class={`status ${st}`}>{#if st === "preset"}<Icon name="alert" size={11} />{/if}{STATUS[st]}</span>
              </div>
              <div class="share" style={`--h:${hue}`} title={`${Math.round(share * 100)}% of the talking`}>
                <div style={`width:${share * 100}%`}></div>
              </div>
              <div class="meta tabular">
                {[`${Math.round(share * 100)}% of talk`, fmtTime(s.talkSeconds || 0), s.referenceSeconds > 0 ? `${Math.round(s.referenceSeconds)} s reference` : "", mode].filter(Boolean).join(" · ")}
              </div>
              {#if note}<div class="stock">{note}</div>{/if}
            </div>
            <div class="pick">
              <button
                id={switchId(s.id)}
                class="switch"
                role="switch"
                aria-checked={s.usePreset}
                aria-label={`Use preset voice for ${s.label}`}
                title={s.usePreset ? "Using a stock Telugu voice. Switch off to use their cloned voice again." : "Use a stock Telugu voice instead of their cloned voice"}
                onclick={() => flip(s)}
                onkeydown={(e) => ownKeys(e)}
              ><span class="track"><span class="thumb"></span></span></button>
              <span class="pick-l" aria-hidden="true">Preset</span>
            </div>
          </div>
          {#if confirming === s.id}
            <div class="confirm" role="alertdialog" tabindex="-1" aria-labelledby={`confirm-t-${s.id}`} aria-describedby={`confirm-b-${s.id}`} onkeydown={(e) => ownKeys(e, s.id)}>
              <p class="c-title" id={`confirm-t-${s.id}`}>Use the preset voice for {s.label}?</p>
              <p class="c-body" id={`confirm-b-${s.id}`}>{presetConfirmBody(st, preparedAhead(app.units, s.id, app.time, "cloned"))}</p>
              <div class="c-actions">
                <button class="btn" use:focusOnMount onclick={() => settle(s.id, false)}>{st === "cloned" ? "Keep cloned voice" : "Cancel"}</button>
                <button class="btn warn" onclick={() => settle(s.id, true)}>Use preset voice</button>
              </div>
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  </section>

  {#if app.current}
    {@const cur = app.current}
    {@const who = app.speakers.find((x) => x.id === cur.speaker)}
    <section class="card now" class:preset={cur.voice === "preset"}>
      <header>
        <h2>Now speaking</h2>
        {#if cur.voice === "preset"}
          <span class="status preset"><Icon name="alert" size={11} />Preset voice</span>
        {:else}
          <span class="status cloned">{clonedLabel(cur, who)}</span>
        {/if}
      </header>
      <div class="who" style={`--h:${speakerHue(cur.speaker)}`}><span class="swatch" aria-hidden="true"></span>{who?.label ?? cur.speaker}</div>
      <p class="te line">{cur.telugu}</p>
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
  .speakers { list-style: none; margin: 0 -8px; padding: 0; display: flex; flex-direction: column; gap: 6px; }
  .speakers li { border-radius: var(--r-md); border: 1px solid transparent; transition: background 0.2s var(--ease), border-color 0.2s var(--ease); }
  /* A speaker on the stock voice is never quiet about it. */
  .speakers li.preset { background: rgb(247 183 51 / 0.07); border-color: rgb(247 183 51 / 0.3); }
  .row { display: grid; grid-template-columns: 40px 1fr auto; align-items: center; gap: 12px; padding: 7px 8px; }
  .avatar { position: relative; width: 40px; height: 40px; display: grid; place-items: center; font-weight: 700; color: hsl(var(--h) 90% 72%); }
  .avatar span { position: relative; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; background: hsl(var(--h) 70% 50% / 0.16); font-size: 13px; }
  .avatar svg { position: absolute; inset: 0; transform: rotate(-90deg); }
  .ring-bg { fill: none; stroke: rgb(255 255 255 / 0.08); stroke-width: 2.5; }
  /* The ring tells the voice's state: a sweeping arc while cloning, closed once cloned. */
  .ring { fill: none; stroke: hsl(var(--h) 85% 62%); stroke-width: 2.5; stroke-linecap: round; stroke-dasharray: 0 100.5; transition: stroke-dasharray 0.5s var(--ease); }
  .avatar.found .ring { stroke-dasharray: 12 100.5; opacity: 0.6; }
  .avatar.cloning svg { animation: spin 1.2s linear infinite; }
  .avatar.cloning .ring { stroke-dasharray: 34 100.5; }
  .avatar.cloned .ring { stroke-dasharray: 100.5 100.5; }
  .avatar.preset .ring { stroke-dasharray: 100.5 100.5; stroke: var(--text-3); opacity: 0.5; }
  .info { min-width: 0; }
  .name-row { display: flex; align-items: center; gap: 8px; }
  .name { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .status {
    flex: none; font-size: 10.5px; font-weight: 650; letter-spacing: 0.02em; padding: 1px 7px; border-radius: 99px;
    border: 1px solid var(--border); color: var(--text-2); background: var(--surface);
  }
  .status.cloning { color: var(--turmeric); border-color: rgb(247 183 51 / 0.35); background: rgb(247 183 51 / 0.1); }
  .status.cloned { color: var(--ok); border-color: rgb(61 220 151 / 0.35); background: rgb(61 220 151 / 0.1); }
  .status { display: inline-flex; align-items: center; gap: 4px; }
  .status.preset { color: #1f1504; border-color: transparent; background: var(--warn); }
  .share { height: 4px; margin: 7px 0 4px; border-radius: 99px; background: rgb(255 255 255 / 0.07); overflow: hidden; }
  .share div { height: 100%; border-radius: 99px; background: hsl(var(--h) 80% 60%); transition: width 0.5s var(--ease); }
  .meta { font-size: 11.5px; color: var(--text-3); }
  .scan { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: var(--text-2); }
  .scan.busy { color: var(--turmeric); }
  .scan.ok { color: var(--ok); }
  .scan.warn, .warn-text { color: var(--warn-ink); }
  .match { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 8px; margin: -4px 0 12px; font-size: 12px; }
  .match-mode { font-weight: 620; color: var(--text); }
  .match-note { color: var(--text-3); }
  .match-note.warn-text { color: var(--warn-ink); }
  .stock { margin-top: 3px; font-size: 11.5px; font-weight: 560; color: var(--warn-ink); }
  .scan-meter { height: 3px; margin: -4px 0 14px; }
  .rate { margin-top: 7px; }
  .whole { margin-top: 10px; }
  .whole[aria-pressed="true"] { border-color: rgb(247 183 51 / 0.55); color: var(--warn-ink); }
  .pick { display: flex; flex-direction: column; align-items: center; gap: 3px; }
  .pick-l { font-size: 10px; font-weight: 600; letter-spacing: 0.03em; color: var(--text-3); }
  li.preset .pick-l { color: var(--warn-ink); }
  .switch { padding: 0; border: 0; background: transparent; cursor: pointer; border-radius: 99px; }
  .switch:focus-visible { outline: 2px solid var(--turmeric); outline-offset: 2px; }
  .track { display: block; width: 34px; height: 20px; border-radius: 99px; background: var(--surface-strong); border: 1px solid var(--border); position: relative; transition: background 0.2s; }
  .thumb { position: absolute; top: 2px; left: 2px; width: 14px; height: 14px; border-radius: 50%; background: var(--text-2); transition: transform 0.2s var(--ease), background 0.2s; }
  .switch[aria-checked="true"] .track { background: var(--warn); border-color: transparent; }
  .switch[aria-checked="true"] .thumb { transform: translateX(14px); background: #1f1504; }
  .confirm {
    margin: 0 8px 8px; padding: 11px 12px; border-radius: var(--r-md); background: var(--bg-raised);
    border: 1px solid rgb(247 183 51 / 0.45); box-shadow: 0 12px 30px -16px rgb(0 0 0 / 0.6); animation: rise 0.2s var(--ease);
  }
  .c-title { margin: 0; font-weight: 620; font-size: 13px; }
  .c-body { margin: 4px 0 10px; font-size: 12px; line-height: 1.5; color: var(--text-2); }
  .c-actions { display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
  .btn {
    height: 30px; padding: 0 12px; border-radius: 999px; cursor: pointer; font-size: 12px; font-weight: 620;
    color: var(--text); background: var(--surface-strong); border: 1px solid var(--border-strong);
  }
  .btn:hover { background: var(--surface-hover); }
  .btn:focus-visible { outline: 2px solid var(--turmeric); outline-offset: 2px; }
  .btn.warn { color: #1f1504; background: var(--warn); border-color: transparent; }
  .btn.warn:hover { filter: brightness(1.06); }
  .now { transition: border-color 0.2s var(--ease); }
  .now.preset { border-color: rgb(247 183 51 / 0.45); }
  .now header { align-items: center; }
  .who { display: flex; align-items: center; gap: 6px; margin: -4px 0 6px; font-size: 12px; font-weight: 600; color: var(--text-2); }
  .swatch { width: 8px; height: 8px; border-radius: 50%; background: hsl(var(--h) 80% 60%); }
  .now .line { margin: 0; font-size: 17px; line-height: 1.6; }
  .note { display: flex; gap: 6px; align-items: center; margin: 0 4px; font-size: 12px; color: var(--text-3); }
  .note :global(svg) { color: var(--turmeric); flex: none; }
  @keyframes pulse { 50% { box-shadow: 0 0 0 7px rgb(247 183 51 / 0.05); } }
  @keyframes spin { to { transform: rotate(270deg); } from { transform: rotate(-90deg); } }
  @keyframes rise { from { opacity: 0; transform: translateY(-4px); } }
</style>
