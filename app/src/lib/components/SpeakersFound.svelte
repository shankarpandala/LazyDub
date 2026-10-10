<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import { hear } from "../hear.svelte";
  import { fmtTime, speakerHue } from "../format";
  import { foundText, speakerChangeCost, voiceSwitchCost } from "../jobs";
  import { SPEAKER_OPTIONS } from "../settings";
  import type { ClientMessage, Render, SpeakersFound } from "../types";

  /**
   * The speaker check (OFFLINE-RENDER §2.3, §5): who the whole-video diarization found, how much and where each talks,
   * their voices, and a correction while it is still free. It never stops the job (M3).
   */
  let { found, job, presets, send }: {
    found: SpeakersFound; job: Render | null; presets: readonly string[]; send: (m: ClientMessage) => void;
  } = $props();

  const text = $derived(foundText(found));
  const nativeVoice = $derived(app.voiceMode === "native");
  const previousVoice = $derived(found.voiceCompatible === false || (nativeVoice && found.voiceMode === "cloned"));
  const voicesReady = $derived(job?.stages.some((s) => s.key === "voices" && s.state === "done") ?? false);
  let fixing = $state(false);
  let count = $state<"auto" | number>("auto");
  $effect.pre(() => {
    count = found.mode === "hint" ? found.speakers.length : "auto";
  });
  let confirmVoice = $state<string | null>(null);
  /** What the stock-voice switch being confirmed would turn it to. */
  let pendingOn = $state(false);

  function listen(sid: string) {
    if (hear.playing === sid) return hear.stop();
    hear.ask(sid);
    send({ type: "voice_sample", videoId: found.videoId, speaker: sid });
  }

  function rerun() {
    app.setSpeakers(found.videoId, count, send);
    fixing = false;
  }

  function setVoice(sid: string, on: boolean) {
    app.setVoice(found.videoId, sid, on, send);
    confirmVoice = null;
  }

  const strip = (activity: readonly number[]) => {
    const max = Math.max(0.0001, ...activity);
    return activity.map((x) => Math.min(1, x / max));
  };
</script>

<section class="card">
  <header>
    <h3>{text.title}</h3>
    <span class="muted">{found.mode === "hint" ? "the count you chose" : "counted over the whole video"}</span>
  </header>
  {#if nativeVoice}
    <p class="muted">{previousVoice
      ? "This saved dub uses its previous voice. New dubs use natural Telugu speech without cloning source voices; the automatic voice can vary between lines."
      : "Natural Telugu speech without cloning source voices. The automatic voice can vary between lines."}</p>
  {/if}
  {#each text.merges as m (m)}<p class="merge"><Icon name="check" size={12} /> {m}</p>{/each}

  <ul class="speakers">
    {#each found.speakers as s (s.id)}
      {@const hue = speakerHue(s.id)}
      {@const stock = !nativeVoice && presets.includes(s.id)}
      <li style={`--h:${hue}`} class:stock>
        <div class="row">
          <div class="avatar" aria-hidden="true"><span>{s.label.replace("Speaker ", "")}</span></div>
          <div class="info">
            <div class="name-row">
              <span class="name">{s.label}</span>
              <span class="talk tabular">{fmtTime(s.talkSeconds)} · {Math.round(s.share * 100)} %</span>
            </div>
            <div class="share"><div style={`width:${Math.min(100, s.share * 100)}%`}></div></div>
            <div class="strip" title={`When ${s.label} talks, over the whole video`}>
              {#each strip(s.activity) as a, i (i)}<span style={`--a:${a}`}></span>{/each}
            </div>
          </div>
          <div class="acts">
            {#if voicesReady && !stock && !previousVoice}
              <button class="btn" aria-pressed={hear.playing === s.id} onclick={() => listen(s.id)} disabled={hear.waiting === s.id}>
                <Icon name={hear.playing === s.id ? "stop" : "volume"} size={13} />
                {hear.waiting === s.id ? "Loading…" : hear.playing === s.id ? "Stop" : "Hear voice"}
              </button>
            {/if}
            {#if nativeVoice}
              <span class="muted">{previousVoice ? "Previous voice" : "Natural Telugu speech"}</span>
            {:else}
              <button class="switch" role="switch" aria-checked={stock} aria-label={`Stock voice for ${s.label}`}
                    title={stock ? "Uses a stock Telugu voice" : "Uses a clone of their voice"}
                    onclick={() => { confirmVoice = s.id; pendingOn = !stock; }}>
              <span class="track"><span class="thumb"></span></span>
              <span class="sl">Stock voice</span>
              </button>
            {/if}
          </div>
        </div>
        {#if !nativeVoice && confirmVoice === s.id}
          <div class="confirm" role="alertdialog" aria-labelledby={`v-${s.id}`}>
            <p class="c-title" id={`v-${s.id}`}>{pendingOn ? `Use a stock Telugu voice for ${s.label}?` : `Use ${s.label}'s own voice again?`}</p>
            <p class="c-body">{voiceSwitchCost(job, s.label)}</p>
            <div class="c-actions">
              <button class="btn" onclick={() => (confirmVoice = null)}>Cancel</button>
              <button class="btn primary" onclick={() => setVoice(s.id, pendingOn)}>{pendingOn ? "Use stock voice" : "Use their voice"}</button>
            </div>
          </div>
        {/if}
      </li>
    {/each}
  </ul>

  <div class="fix">
    {#if !fixing}
      <button class="link" onclick={() => (fixing = true)}>Wrong count?</button>
      <span class="muted">{speakerChangeCost(job)}</span>
    {:else}
      <div class="fix-row">
        <div class="seg" role="radiogroup" aria-label="How many people speak">
          {#each SPEAKER_OPTIONS as k (k)}
            <button role="radio" aria-checked={count === k} class:on={count === k} onclick={() => (count = k)}>{k === "auto" ? "Auto" : k}</button>
          {/each}
        </div>
        <button class="btn primary" onclick={rerun}><Icon name="redo" size={13} /> Re-run speakers</button>
        <button class="btn" onclick={() => (fixing = false)}>Cancel</button>
      </div>
      <p class="muted">{speakerChangeCost(job)}</p>
    {/if}
  </div>
</section>

<style>
  .card { padding: 18px; border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur); }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 8px; }
  h3 { margin: 0; font-size: 15px; font-weight: 650; }
  .muted { color: var(--text-3); font-size: 12px; }
  .merge { display: flex; align-items: center; gap: 6px; margin: 0 0 6px; font-size: 12.5px; color: var(--text-2); }
  .merge :global(svg) { color: var(--ok); }
  .speakers { list-style: none; margin: 10px -8px 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
  li { border-radius: var(--r-md); border: 1px solid transparent; }
  li.stock { background: rgb(247 183 51 / 0.07); border-color: rgb(247 183 51 / 0.3); }
  .row { display: grid; grid-template-columns: 38px minmax(0, 1fr) auto; gap: 12px; align-items: center; padding: 8px; }
  .avatar { width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center; background: hsl(var(--h) 70% 50% / 0.16); box-shadow: inset 0 0 0 2px hsl(var(--h) 85% 62%); color: hsl(var(--h) 90% 72%); font-weight: 700; }
  :global([data-theme="light"]) .avatar { color: hsl(var(--h) 70% 38%); }
  .info { min-width: 0; }
  .name-row { display: flex; justify-content: space-between; gap: 8px; }
  .name { font-weight: 620; }
  .talk { color: var(--text-3); font-size: 12px; }
  .share { height: 4px; margin: 6px 0 5px; border-radius: 99px; background: var(--surface-strong); overflow: hidden; }
  .share div { height: 100%; border-radius: 99px; background: hsl(var(--h) 80% 60%); transition: width 0.5s var(--ease); }
  .strip { display: grid; grid-template-columns: repeat(120, 1fr); gap: 1px; height: 14px; align-items: end; }
  .strip span { height: calc(2px + var(--a) * 12px); border-radius: 1px; background: hsl(var(--h) 80% 60% / calc(0.15 + var(--a) * 0.85)); }
  .acts { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
  .btn {
    display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 11px; border-radius: 999px; cursor: pointer;
    font-size: 12px; font-weight: 600; color: var(--text); background: var(--surface-strong); border: 1px solid var(--border-strong); white-space: nowrap;
  }
  .btn:hover:not(:disabled) { background: var(--surface-hover); }
  .btn:disabled { opacity: 0.6; cursor: default; }
  .btn.primary { background: var(--accent-grad); color: #1a0f08; border-color: transparent; }
  .switch { display: inline-flex; align-items: center; gap: 6px; padding: 0; border: 0; background: none; cursor: pointer; font-size: 11.5px; color: var(--text-3); border-radius: 99px; }
  .track { display: block; width: 30px; height: 18px; border-radius: 99px; background: var(--surface-strong); border: 1px solid var(--border); position: relative; transition: background 0.2s; }
  .thumb { position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; border-radius: 50%; background: var(--text-2); transition: transform 0.2s var(--ease); }
  .switch[aria-checked="true"] .track { background: var(--warn); border-color: transparent; }
  .switch[aria-checked="true"] .thumb { transform: translateX(12px); background: #1f1504; }
  li.stock .sl { color: var(--warn-ink); font-weight: 600; }
  .confirm { margin: 0 8px 8px; padding: 10px 12px; border-radius: var(--r-md); background: var(--bg-raised); border: 1px solid rgb(247 183 51 / 0.45); }
  .c-title { margin: 0; font-weight: 620; font-size: 13px; }
  .c-body { margin: 4px 0 10px; font-size: 12px; color: var(--text-2); }
  .c-actions { display: flex; justify-content: flex-end; gap: 8px; }
  .fix { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
  .fix-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .fix .muted { margin: 0; }
  .link { border: 0; background: none; color: var(--vermilion); font-weight: 600; cursor: pointer; padding: 0; }
  .link:hover { text-decoration: underline; }
  .seg { display: inline-flex; gap: 3px; padding: 3px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border); }
  .seg button { min-width: 32px; height: 28px; padding: 0 10px; border: 0; border-radius: 9px; background: transparent; color: var(--text-2); font-weight: 600; cursor: pointer; }
  .seg button.on { background: var(--accent-grad); color: #1a0f08; }
</style>
