<script lang="ts">
  import Icon from "./Icon.svelte";
  import SpeakersFound from "./SpeakersFound.svelte";
  import StatusChip from "./StatusChip.svelte";
  import Thumb from "./Thumb.svelte";
  import { app } from "../state.svelte";
  import { engineAsset } from "../engine";
  import { fmtBytes, fmtDuration, fmtTime } from "../format";
  import { chip, coverageShares, elapsedText, etaText, folderText, groupRows, isActive, live, sleptText } from "../jobs";
  import type { ClientMessage } from "../types";

  let { send }: { send: (m: ClientMessage) => void } = $props();

  const job = $derived(app.job!);
  const item = $derived(app.jobItem);
  const rows = $derived(groupRows(job.stages));
  const running = $derived(job.status === "running" || job.status === "waiting");
  const queued = $derived(job.status === "queued" || (job.position != null && !running));
  let now = $state(Date.now());
  $effect(() => {
    const id = setInterval(() => (now = Date.now()), 15_000);
    return () => clearInterval(id);
  });
  const resetsAt = $derived(app.claudeProblem?.videoId === job.videoId ? app.claudeProblem.resetsAt ?? null : null);
  const c = $derived(chip(item ? live(item, job) : { ...job, duration: 0, expired: false }, now, resetsAt));
  const slept = $derived(sleptText(job.slept, now));
  const out = $derived(job.output);
  const preview = $derived(job.status === "done" && (out?.kind === "preview" || job.stopAt != null));
  const shares = $derived(coverageShares(job.coverage));
  const name = (p: string) => p.split(/[\\/]/).pop() ?? p;
  const dir = (p: string) => p.replace(/[\\/][^\\/]*$/, "");
  const presets = $derived(item?.settings?.presets ?? []);

  function again() {
    if (item) app.continueJob(item, send);
  }
</script>

<section class="jobview">
  <button class="back" onclick={() => app.home()}><Icon name="back" size={16} /> Your dubs</button>

  <header class="card head">
    <Thumb src={item?.thumb ? engineAsset(item.thumb) : null} duration={item?.duration ?? 0} title={item?.title} />
    <div class="about">
      <h2>{item?.title || job.videoId}</h2>
      <div class="meta">{item?.channel ?? ""}{job.stopAt ? ` · first ${Math.round(job.stopAt / 60)} minutes` : ""}</div>
      <StatusChip chip={c} />
      <div class="times tabular">
        {#if running && job.eta != null}<span class="eta">{etaText(job.eta)}</span>{/if}
        {#if job.elapsed}<span>{elapsedText(job.elapsed, job.status === "done")}</span>{/if}
      </div>
    </div>
    <div class="head-acts">
      {#if running || queued}
        <button class="btn" onclick={() => app.pause(job.videoId, send)}><Icon name="pause" size={12} /> Pause</button>
      {:else if job.status !== "done" || item?.expired}
        <button class="btn primary" onclick={again}><Icon name="play" size={12} /> {item?.expired ? "Dub again" : job.status === "failed" ? "Try again" : "Resume"}</button>
      {/if}
    </div>
  </header>

  {#if slept}
    <p class="note"><Icon name="clock" size={14} /> {slept}. Keep the lid open and the Mac plugged in; the screen can turn off.</p>
  {/if}
  {#if job.status === "failed" && job.error}
    <div class="card failed" role="alert"><Icon name="alert" size={16} /> <span>{job.error}</span></div>
  {/if}

  <div class="grid">
    <section class="card steps" aria-label="Progress">
      <ol>
        {#each rows as r (r.label)}
          <li class={r.state}>
            <span class="bullet" aria-hidden="true">
              {#if r.state === "done"}<Icon name="check" size={12} />{:else if r.state === "failed"}<Icon name="alert" size={11} />{/if}
            </span>
            <div class="r-main">
              <div class="r-top">
                <span class="r-label">{r.label}</span>
                <span class="r-count tabular">{r.count}</span>
              </div>
              {#if r.state === "running"}
                <div class="bar" class:indeterminate={r.indeterminate}>
                  <div style={r.indeterminate ? "" : `width:${(r.share ?? 0) * 100}%`}></div>
                </div>
              {/if}
              {#if r.note && r.state === "running"}<div class="r-note">{r.note}</div>{/if}
            </div>
            <span class="r-time tabular">
              {#if r.state === "running" && running && r.eta != null}{etaText(r.eta)}{:else if r.state === "done" && r.seconds >= 1}{fmtDuration(r.seconds)}{/if}
            </span>
          </li>
        {/each}
      </ol>
    </section>

    <div class="side">
      {#if job.status === "done" && out}
        <section class="card done" class:missing={out.missing}>
          {#if out.missing}
            <h3><Icon name="alert" size={16} /> File moved or deleted</h3>
            <p class="muted">{name(out.path)} isn't in {folderText(dir(out.path))} any more.
              {out.kind === "whole" ? "Saving it again downloads the video and separates the background sound again first." : ""}</p>
            <div class="row">
              <button class="btn" onclick={() => app.openOutput(job.videoId, true, send)}><Icon name="folder" size={14} /> Show folder</button>
              <button class="btn primary" onclick={() => app.resume(job.videoId, send)}><Icon name="redo" size={14} /> Save again</button>
            </div>
          {:else}
            <h3><Icon name="film" size={16} /> {preview ? "Preview saved" : "Saved"}</h3>
            <p class="file">{name(out.path)}</p>
            <p class="muted">{fmtBytes(out.bytes)} · {folderText(dir(out.path))}</p>
            <div class="row">
              <button class="btn primary" onclick={() => app.openOutput(job.videoId, false, send)}><Icon name="play" size={13} /> Open video</button>
              <button class="btn" onclick={() => app.openOutput(job.videoId, true, send)}><Icon name="folder" size={14} /> Show in Finder</button>
            </div>
            <p class="what"><Icon name="music" size={13} /> {out.bed ? "Telugu voices over the original music and sounds" : "Telugu voices"} · <Icon name="subtitles" size={13} /> Telugu and English subtitles</p>
          {/if}
          {#if out.warning}<p class="warn"><Icon name="alert" size={13} /> {out.warning}</p>{/if}
          {#if preview && item}
            <button class="btn wide" onclick={() => app.continueJob(item, send, true)}><Icon name="arrow" size={14} /> Continue to the whole video</button>
          {/if}
          <dl class="facts">
            {#if shares.length}
              <dt>Meaning kept</dt>
              <dd>
                <div class="cov" aria-hidden="true">{#each shares as s (s.key)}<span class={`c-${s.key}`} style={`flex:${s.share}`}></span>{/each}</div>
                {shares.map((s) => `${s.label} ${Math.round(s.share * 100)} %`).join(" · ")}
              </dd>
            {/if}
            {#if job.report}
              <dt>Lines</dt>
              <dd>{job.report.lines} voiced{job.report.long ? ` · ${job.report.long} still long` : ""}{job.report.unreviewed ? ` · ${job.report.unreviewed} unreviewed` : ""}{job.coverage?.skipped ? ` · ${job.coverage.skipped} skipped` : ""}</dd>
              <dt>Running over</dt>
              <dd>{job.report.overdraft.toFixed(1)} s in all{job.report.carried ? `, ${job.report.carried.toFixed(1)} s carried to the next lines` : ""}</dd>
            {/if}
            {#if out.loudness}
              <dt>Loudness</dt>
              <dd class="tabular">{out.loudness.I.toFixed(1)} LUFS · peak {out.loudness.TP.toFixed(1)} dBTP</dd>
            {/if}
          </dl>
        </section>
      {/if}

      {#if app.found}
        <SpeakersFound found={app.found} {job} {presets} {send} />
      {:else if job.status !== "done" && job.status !== "failed"}
        <section class="card wait">
          <h3>Speakers</h3>
          <p class="muted">Once Maata has listened to the whole video, it shows who speaks. You can hear each voice{app.voiceMode === "native" ? ", choose a Telugu voice for each speaker," : ""} and correct the count.</p>
        </section>
      {/if}

      {#if isActive(job.status)}
        <p class="note small"><Icon name="clock" size={13} /> Maata keeps dubbing while its window is closed. Quitting Maata pauses; it continues when you open it again.</p>
      {/if}
    </div>
  </div>
</section>

<style>
  .jobview { max-width: 1180px; margin: 0 auto; padding: 18px 32px 40px; display: flex; flex-direction: column; gap: 14px; }
  .back { align-self: flex-start; display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; margin-left: -10px; border: 0; border-radius: 10px; background: none; color: var(--text-2); cursor: pointer; font-weight: 560; }
  .back:hover { color: var(--text); background: var(--surface-hover); }
  .card { padding: 18px; border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur); }
  .head { display: grid; grid-template-columns: 220px minmax(0, 1fr) auto; gap: 20px; align-items: center; animation: rise 0.35s var(--ease) both; }
  .about { display: flex; flex-direction: column; align-items: flex-start; gap: 7px; min-width: 0; }
  h2 { margin: 0; font-size: 20px; line-height: 1.25; letter-spacing: -0.01em; font-weight: 640; }
  .meta { color: var(--text-2); }
  .times { display: flex; gap: 12px; color: var(--text-3); font-size: 12.5px; }
  .times .eta { color: var(--text); font-weight: 600; }
  .head-acts { display: flex; gap: 8px; }
  .note { display: flex; align-items: center; gap: 8px; margin: 0; color: var(--text-2); font-size: 13px; }
  .note.small { font-size: 12px; color: var(--text-3); }
  .note :global(svg) { color: var(--turmeric); flex: none; }
  .failed { display: flex; gap: 10px; align-items: flex-start; border-color: rgb(255 93 93 / 0.4); background: rgb(255 93 93 / 0.07); color: var(--text); }
  .failed :global(svg) { color: var(--danger); flex: none; margin-top: 2px; }

  .grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 14px; align-items: start; }
  .side { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .steps ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }
  .steps li { position: relative; display: grid; grid-template-columns: 22px minmax(0, 1fr) auto; gap: 12px; align-items: start; padding: 9px 0; color: var(--text-3); }
  /* The rail between the bullets. */
  .steps li:not(:last-child)::after { content: ""; position: absolute; left: 10px; top: 31px; bottom: -7px; width: 2px; border-radius: 2px; background: var(--border); }
  .steps li.done:not(:last-child)::after { background: rgb(61 220 151 / 0.45); }
  .bullet { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid var(--border-strong); display: grid; place-items: center; background: var(--bg); position: relative; z-index: 1; }
  li.done, li.running, li.failed { color: var(--text); }
  li.done .bullet { background: var(--ok); border-color: var(--ok); color: #06150e; }
  li.running .bullet { border-color: var(--turmeric); box-shadow: 0 0 0 4px rgb(247 183 51 / 0.15); animation: pulse 1.4s ease-in-out infinite; }
  li.failed .bullet { background: var(--danger); border-color: var(--danger); color: #1f0505; }
  .r-main { min-width: 0; }
  .r-top { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
  .r-label { font-weight: 600; }
  li.todo .r-label { font-weight: 520; }
  .r-count { font-size: 12.5px; color: var(--text-2); white-space: nowrap; }
  .r-note { margin-top: 4px; font-size: 12px; color: var(--warn-ink); }
  .r-time { font-size: 12px; color: var(--text-3); white-space: nowrap; min-width: 92px; text-align: right; padding-top: 1px; }
  li.running .r-time { color: var(--text-2); }
  .bar { margin-top: 7px; height: 4px; border-radius: 99px; background: var(--surface-strong); overflow: hidden; position: relative; }
  .bar div { height: 100%; border-radius: 99px; background: var(--accent-grad); transition: width 0.5s var(--ease); }
  .bar.indeterminate div { position: absolute; width: 35%; animation: slide 1.3s ease-in-out infinite; }

  .done h3, .wait h3 { display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 15px; }
  .done h3 :global(svg) { color: var(--ok); }
  .done.missing { border-color: rgb(247 183 51 / 0.45); }
  .done.missing h3 :global(svg) { color: var(--warn-ink); }
  .file { margin: 0; font-weight: 600; word-break: break-word; user-select: text; -webkit-user-select: text; }
  .muted { margin: 2px 0 12px; color: var(--text-3); font-size: 12.5px; }
  .what { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; color: var(--text-3); font-size: 12px; }
  .what :global(svg) { color: var(--turmeric); }
  .warn { display: flex; gap: 6px; align-items: center; margin: 10px 0 0; color: var(--warn-ink); font-size: 12.5px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .wide { margin-top: 12px; }
  .facts { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 14px; margin: 14px 0 0; padding-top: 12px; border-top: 1px solid var(--border); font-size: 12.5px; }
  dt { color: var(--text-3); }
  dd { margin: 0; color: var(--text-2); }
  .cov { display: flex; height: 5px; border-radius: 99px; overflow: hidden; margin: 5px 0 4px; gap: 1px; }
  .c-C { background: var(--ok); }
  .c-m { background: var(--turmeric); }
  .c-P { background: var(--vermilion); }
  .c-E { background: var(--danger); }
  .btn {
    display: inline-flex; align-items: center; gap: 6px; height: 34px; padding: 0 14px; border-radius: 999px; cursor: pointer;
    font-size: 13px; font-weight: 600; color: var(--text); background: var(--surface-strong); border: 1px solid var(--border-strong); white-space: nowrap;
  }
  .btn:hover { background: var(--surface-hover); }
  .btn.primary { background: var(--accent-grad); color: #1a0f08; border-color: transparent; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } .head { grid-template-columns: 160px minmax(0, 1fr); } .head-acts { grid-column: 1 / -1; } }
  @keyframes pulse { 50% { box-shadow: 0 0 0 7px rgb(247 183 51 / 0.05); } }
  @keyframes slide { from { left: -35%; } to { left: 100%; } }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
