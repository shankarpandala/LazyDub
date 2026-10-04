<script lang="ts">
  import Icon from "./Icon.svelte";
  import StatusChip from "./StatusChip.svelte";
  import Thumb from "./Thumb.svelte";
  import { app } from "../state.svelte";
  import { fmtTime } from "../format";
  import { chip, estimateLine, folderText, isActive, jobsAhead, live } from "../jobs";
  import {
    PREVIEW_AT, SPEAKER_OPTIONS, SPEED_CAP_MAX, SPEED_CAP_MIN, SPEED_CAP_STEP, STYLE_OPTIONS, TTS_SCRIPT_OPTIONS, clampSpeedCap,
  } from "../settings";
  import type { ClientMessage, DubOptions } from "../types";

  let { send }: { send: (m: ClientMessage) => void } = $props();

  const video = $derived(app.video!);
  // The options start from the last job's (§5) each time another video opens.
  let opts = $state<DubOptions>({ ...app.lastOptions });
  let more = $state(false);
  let shownFor = "";
  $effect.pre(() => {
    if (video.videoId !== shownFor) {
      shownFor = video.videoId;
      opts = { ...app.lastOptions };
      more = false;
    }
  });
  const job = $derived(app.videoJob);
  const ahead = $derived(jobsAhead(app.jobs, video.videoId));
  const short = $derived(video.duration > 0 && video.duration <= PREVIEW_AT);
  const line = $derived(estimateLine(video.estimate, {
    stopAt: short ? null : opts.stopAt, outputDir: app.settings?.outputDir ?? null, ahead, claude: app.claude !== null,
  }));
  const status = $derived(job ? app.renders[job.videoId]?.status ?? job.status : null);
  const preview = $derived(!!job && status === "done" && (job.output?.kind === "preview" || job.stopAt != null));

  function dub() {
    app.dub({ ...opts, stopAt: short ? null : opts.stopAt, speedCap: clampSpeedCap(opts.speedCap) }, send);
  }
</script>

<section class="new">
  <button class="back" onclick={() => app.home()}><Icon name="back" size={16} /> Your dubs</button>

  <div class="card">
    <div class="video">
      <Thumb src={video.thumbnail} duration={video.duration} title={video.title} />
      <div class="about">
        <h2>{video.title || video.videoId}</h2>
        <div class="meta">{video.channel}{video.duration ? ` · ${fmtTime(video.duration)}` : ""}</div>
        {#if job}
          <StatusChip chip={chip(live(job, app.renders[job.videoId]))} />
        {/if}
      </div>
    </div>

    {#if job}
      <div class="existing">
        <p class="muted">
          {#if isActive(status ?? "paused")}This video is being dubbed.{:else if status === "done"}This video is dubbed already.{:else}This video has a dub started.{/if}
        </p>
        <div class="row">
          {#if status === "done" && job.output && !job.output.missing}
            <button class="btn primary" onclick={() => app.openOutput(job.videoId, false, send)}><Icon name="play" size={13} /> Open video</button>
            <button class="btn" onclick={() => app.openOutput(job.videoId, true, send)}><Icon name="folder" size={14} /> Show in Finder</button>
          {/if}
          {#if preview}
            <button class="btn" onclick={() => { app.continueJob(job, send, true); app.openJob(job.videoId); }}>
              <Icon name="arrow" size={14} /> Continue to the whole video
            </button>
          {:else if !isActive(status ?? "paused") && status !== "done" || job.expired}
            <button class="btn primary" onclick={() => { app.continueJob(job, send); app.openJob(job.videoId); }}>
              <Icon name="play" size={12} /> Continue
            </button>
          {/if}
          <button class="btn" onclick={() => app.openJob(job.videoId)}><Icon name="clock" size={14} /> Progress</button>
        </div>
      </div>
    {:else}
      <div class="options">
        <div class="opt">
          <span class="name" id="o-speakers">Speakers</span>
          <div class="seg" role="radiogroup" aria-labelledby="o-speakers">
            {#each SPEAKER_OPTIONS as k (k)}
              <button role="radio" aria-checked={opts.speakers === k} class:on={opts.speakers === k} onclick={() => (opts.speakers = k)}>
                {k === "auto" ? "Auto" : k}
              </button>
            {/each}
          </div>
          <span class="hint">{opts.speakers === "auto" ? "Maata counts them over the whole video; you can correct it once it has." : "Exactly this many people speak."}</span>
        </div>

        <div class="opt">
          <span class="name" id="o-style">Style</span>
          <div class="seg" role="radiogroup" aria-labelledby="o-style">
            {#each STYLE_OPTIONS as o (o.id)}
              <button role="radio" aria-checked={opts.style === o.id} class:on={opts.style === o.id} onclick={() => (opts.style = o.id)}>{o.label}</button>
            {/each}
          </div>
          <span class="hint">{STYLE_OPTIONS.find((o) => o.id === opts.style)?.hint}</span>
        </div>

        <div class="opt">
          <span class="name" id="o-length">Length</span>
          <div class="seg" role="radiogroup" aria-labelledby="o-length">
            <button role="radio" aria-checked={short || opts.stopAt === null} class:on={short || opts.stopAt === null} onclick={() => (opts.stopAt = null)}>Whole video</button>
            <button role="radio" aria-checked={!short && opts.stopAt !== null} class:on={!short && opts.stopAt !== null} disabled={short}
                    onclick={() => (opts.stopAt = PREVIEW_AT)}>First 15 minutes</button>
          </div>
          <span class="hint">{short ? "The video is 15 minutes or shorter." : opts.stopAt ? "A preview to judge the voices and the Telugu; it can continue to the whole video later." : "The whole video, in one file."}</span>
        </div>

        <details bind:open={more}>
          <summary>More options</summary>
          <div class="opt">
            <label class="name" for="o-cap">Speed-up</label>
            <div class="slider">
              <input id="o-cap" type="range" min={SPEED_CAP_MIN} max={SPEED_CAP_MAX} step={SPEED_CAP_STEP} bind:value={opts.speedCap} />
              <span class="tabular">up to {clampSpeedCap(opts.speedCap).toFixed(2)}×</span>
            </div>
            <span class="hint">How much a Telugu line may be sped up to fit its time. Above about 1.2× speech sounds rushed.</span>
          </div>
          <div class="opt">
            <span class="name" id="o-script">English words</span>
            <div class="radios" role="radiogroup" aria-labelledby="o-script">
              {#each TTS_SCRIPT_OPTIONS as o (o.id)}
                <label class="radio">
                  <input type="radio" name="tts-script" value={o.id} bind:group={opts.ttsScript} />
                  <span><span class="r-l">{o.label}</span><span class="hint">{o.hint}</span></span>
                </label>
              {/each}
            </div>
          </div>
        </details>

        <div class="opt folder">
          <span class="name">Saves to</span>
          <span class="path" title={app.settings?.outputDir}><Icon name="folder" size={14} /> {app.settings ? folderText(app.settings.outputDir) : "…"}</span>
          <button class="link" onclick={() => (app.settingsOpen = true)}>Change</button>
        </div>
      </div>

      <div class="go-row">
        <p class="estimate">{line}</p>
        <button class="dub" onclick={dub} disabled={app.connection !== "open"}>
          <Icon name="sparkle" size={16} /> {ahead > 0 ? "Add to queue" : "Dub"}
        </button>
      </div>
    {/if}
  </div>
</section>

<style>
  .new { max-width: 860px; margin: 0 auto; padding: 18px 32px 40px; display: flex; flex-direction: column; gap: 14px; }
  .back { align-self: flex-start; display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; margin-left: -10px; border: 0; border-radius: 10px; background: none; color: var(--text-2); cursor: pointer; font-weight: 560; }
  .back:hover { color: var(--text); background: var(--surface-hover); }
  .card {
    padding: 22px; border-radius: var(--r-xl); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur);
    box-shadow: var(--shadow-lg); animation: rise 0.35s var(--ease) both; display: flex; flex-direction: column; gap: 22px;
  }
  .video { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 20px; align-items: center; }
  .about { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; min-width: 0; }
  h2 { margin: 0; font-size: 21px; line-height: 1.25; letter-spacing: -0.01em; font-weight: 640; }
  .meta { color: var(--text-2); }
  .muted { color: var(--text-2); margin: 0 0 10px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }

  .options { display: flex; flex-direction: column; gap: 16px; }
  .opt { display: grid; grid-template-columns: 110px minmax(0, 1fr); align-items: center; gap: 4px 16px; }
  .opt .hint { grid-column: 2; }
  .name { color: var(--text-2); font-weight: 600; font-size: 13px; }
  .hint { color: var(--text-3); font-size: 12px; }
  .seg { display: inline-flex; flex-wrap: wrap; gap: 4px; padding: 4px; border-radius: 14px; background: var(--surface); border: 1px solid var(--border); justify-self: start; }
  .seg button {
    min-width: 38px; height: 32px; padding: 0 13px; border: 0; border-radius: 10px; background: transparent; color: var(--text-2);
    font-weight: 600; cursor: pointer; transition: background 0.15s var(--ease), color 0.15s var(--ease);
  }
  .seg button:hover:not(:disabled) { color: var(--text); background: var(--surface-hover); }
  .seg button.on { background: var(--accent-grad); color: #1a0f08; box-shadow: 0 4px 16px -6px rgb(242 96 60 / 0.6); }
  .seg button:disabled { opacity: 0.4; cursor: default; }
  details { border-top: 1px solid var(--border); padding-top: 12px; display: flex; flex-direction: column; gap: 14px; }
  details[open] { padding-bottom: 2px; }
  details[open] > .opt { margin-top: 14px; }
  summary { cursor: pointer; color: var(--text-2); font-weight: 600; font-size: 13px; width: max-content; }
  summary:hover { color: var(--text); }
  .slider { display: flex; align-items: center; gap: 12px; }
  .slider input { width: 220px; accent-color: var(--vermilion); }
  .radios { display: flex; flex-direction: column; gap: 8px; }
  .radio { display: flex; gap: 10px; align-items: flex-start; cursor: pointer; }
  .radio input { margin-top: 3px; accent-color: var(--vermilion); }
  .radio > span { display: flex; flex-direction: column; }
  .r-l { font-weight: 560; }
  .folder .path { display: inline-flex; align-items: center; gap: 6px; color: var(--text); font-weight: 560; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .folder { grid-template-columns: 110px auto 1fr; }
  .link { justify-self: start; border: 0; background: none; color: var(--vermilion); font-weight: 600; cursor: pointer; padding: 2px 4px; }
  .link:hover { text-decoration: underline; }

  .go-row { display: flex; align-items: center; gap: 18px; justify-content: space-between; border-top: 1px solid var(--border); padding-top: 18px; }
  .estimate { margin: 0; color: var(--text-2); font-size: 13px; }
  .dub {
    flex: none; display: inline-flex; align-items: center; gap: 8px; height: 46px; padding: 0 26px; border-radius: 999px; border: 0; cursor: pointer;
    background: var(--accent-grad); color: #1a0f08; font-weight: 700; font-size: 15px; box-shadow: var(--accent-glow);
    transition: transform 0.15s var(--ease), filter 0.15s;
  }
  .dub:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.05); }
  .dub:disabled { opacity: 0.5; cursor: default; }
  .btn {
    display: inline-flex; align-items: center; gap: 6px; height: 34px; padding: 0 14px; border-radius: 999px; cursor: pointer;
    font-size: 13px; font-weight: 600; color: var(--text); background: var(--surface-strong); border: 1px solid var(--border-strong);
  }
  .btn:hover { background: var(--surface-hover); }
  .btn.primary { background: var(--accent-grad); color: #1a0f08; border-color: transparent; }
  @media (max-width: 760px) { .video { grid-template-columns: 1fr; } .go-row { flex-direction: column; align-items: stretch; } }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
